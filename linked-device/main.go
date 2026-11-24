package main

import (
	"context"
	"crypto/subtle"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/polymorfa/hypermeow"
	"github.com/polymorfa/hypermeow/proto/waCompanionReg"
	"github.com/polymorfa/hypermeow/store"
	"github.com/polymorfa/hypermeow/store/sqlstore"
	"github.com/polymorfa/hypermeow/types"
	"github.com/polymorfa/hypermeow/types/events"
	waLog "github.com/polymorfa/hypermeow/util/log"
	meow "github.com/purpshell/meowcaller"
	"github.com/skip2/go-qrcode"
	"google.golang.org/protobuf/proto"
	_ "modernc.org/sqlite"
)

type audioQueue struct {
	mu   sync.Mutex
	data []byte
}

func (a *audioQueue) ReadFrame() ([]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]float32, meow.FrameSamples)
	n := min(len(a.data)/2, len(out))
	for i := 0; i < n; i++ {
		out[i] = float32(int16(binary.LittleEndian.Uint16(a.data[i*2:]))) / 32768
	}
	a.data = a.data[n*2:]
	return out, nil
}
func (a *audioQueue) Close() error { return nil }
func (a *audioQueue) add(b []byte) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.data)+len(b) > 64000 {
		a.data = nil
	}
	a.data = append(a.data, b...)
}
func (a *audioQueue) clear() { a.mu.Lock(); a.data = nil; a.mu.Unlock() }

type manager struct {
	db       *pgxpool.Pool
	mu       sync.Mutex
	sessions map[string]*linkedSession
}

type linkedSession struct {
	cancel context.CancelFunc
	done   chan struct{}
	client *whatsmeow.Client
}

func (m *manager) status(id, status, qr, errText string, seconds int) {
	_, err := m.db.Exec(context.Background(), `update portal_whatsapp_links set status=$2,qr=nullif($3,''),error=$4,qr_expires_at=case when $5>0 then now()+$5*interval '1 second' else null end,updated_at=now() where id=$1`, id, status, qr, errText, seconds)
	if err != nil {
		log.Printf("status update failed: %v", err)
	}
}
func (m *manager) run(ctx context.Context, id string) {
	defer func() { m.mu.Lock(); delete(m.sessions, id); m.mu.Unlock() }()
	if err := os.MkdirAll("sessions", 0700); err != nil {
		m.status(id, "error", "", "Cannot create session storage", 0)
		return
	}
	path := filepath.Join("sessions", id+".db")
	container, err := sqlstore.New(ctx, "sqlite", "file:"+path+"?_pragma=foreign_keys(1)&_pragma=busy_timeout(5000)", waLog.Noop)
	if err != nil {
		m.status(id, "error", "", "Cannot open session", 0)
		return
	}
	defer container.Close()
	device, err := container.GetFirstDevice(ctx)
	if err != nil {
		m.status(id, "error", "", "Cannot load session", 0)
		return
	}
	client := whatsmeow.NewClient(device, waLog.Noop)
	m.mu.Lock()
	if session := m.sessions[id]; session != nil {
		session.client = client
	}
	m.mu.Unlock()
	calls := meow.NewClient(client)
	client.AddEventHandler(func(evt any) {
		switch evt.(type) {
		case *events.CallOffer:
			log.Printf("WhatsApp call offer event received for connection %s", id)
		case *events.Connected:
			m.status(id, "connected", "", "", 0)
			_, err := m.db.Exec(context.Background(), `update portal_whatsapp_links set phone=$2 where id=$1`, id, client.Store.GetJID().User)
			if err != nil {
				log.Printf("phone update failed: %v", err)
			}
			go client.SendPresence(context.Background(), types.PresenceAvailable)
		case *events.LoggedOut:
			m.status(id, "error", "", "Device was unlinked in WhatsApp", 0)
		case *events.Disconnected:
			if ctx.Err() == nil {
				m.status(id, "connecting", "", "Reconnecting", 0)
			}
		}
	})
	calls.OnIncomingCall(func(call *meow.Call) {
		log.Printf("WhatsApp call offer accepted by adapter for connection %s", id)
		go m.handleCall(ctx, id, call)
	})
	if client.Store.ID == nil {
		qr, err := client.GetQRChannel(ctx)
		if err != nil {
			m.status(id, "error", "", "Unable to create pairing code", 0)
			return
		}
		if err = client.Connect(); err != nil {
			m.status(id, "error", "", "Unable to connect to WhatsApp", 0)
			return
		}
		for event := range qr {
			if event.Event == "code" {
				image, err := qrcode.Encode(event.Code, qrcode.Medium, 256)
				if err == nil {
					m.status(id, "pairing", base64.StdEncoding.EncodeToString(image), "", int(event.Timeout.Seconds()))
				}
			}
		}
	} else if err = client.Connect(); err != nil {
		m.status(id, "error", "", "Unable to connect to WhatsApp", 0)
		return
	}
	defer client.Disconnect()
	<-ctx.Done()
	m.status(id, "offline", "", "", 0)
}
func (m *manager) handleCall(ctx context.Context, id string, call *meow.Call) {
	var enabled, voiceEnabled bool
	if err := m.db.QueryRow(ctx, `select enabled,voice_enabled from portal_whatsapp_links where id=$1`, id).Scan(&enabled, &voiceEnabled); err != nil {
		log.Printf("incoming call state lookup failed for connection %s: %v", id, err)
		return
	}
	if !enabled || !voiceEnabled {
		log.Printf("incoming call ignored for connection %s: enabled=%t voice_enabled=%t", id, enabled, voiceEnabled)
		return
	}
	secret := os.Getenv("LINKED_BRIDGE_SECRET")
	q := url.Values{"customer": {id}, "call": {call.ID()}, "peer": {call.Peer().User}}
	h := http.Header{}
	h.Set("Authorization", "Bearer "+secret)
	ws, _, err := websocket.Dial(ctx, "ws://127.0.0.1:8098/call?"+q.Encode(), &websocket.DialOptions{HTTPHeader: h})
	if err != nil {
		log.Printf("voice bridge unavailable for connection %s: %v", id, err)
		return
	}
	defer ws.CloseNow()
	if err = call.Answer(); err != nil {
		log.Printf("incoming call answer failed for connection %s: %v", id, err)
		return
	}
	log.Printf("incoming call answered for connection %s", id)
	defer call.Hangup()
	call.OnEnd(func(string) { ws.CloseNow() })
	audio := make(chan []byte, 25)
	var once sync.Once
	done := make(chan struct{})
	stop := func() { once.Do(func() { close(done) }) }
	defer stop()
	call.Receive(meow.SinkFunc(func(frame []float32) {
		data := make([]byte, len(frame)*2)
		for i, v := range frame {
			if v > 1 {
				v = 1
			}
			if v < -1 {
				v = -1
			}
			binary.LittleEndian.PutUint16(data[i*2:], uint16(int16(v*32767)))
		}
		select {
		case audio <- data:
		default:
		}
	}))
	queue := &audioQueue{}
	call.Play(queue)
	go func() {
		for {
			select {
			case b := <-audio:
				if ws.Write(ctx, websocket.MessageBinary, b) != nil {
					ws.CloseNow()
					return
				}
			case <-done:
				return
			case <-ctx.Done():
				return
			}
		}
	}()
	for {
		kind, b, err := ws.Read(ctx)
		if err != nil {
			return
		}
		if kind == websocket.MessageBinary {
			queue.add(b)
		} else if string(b) == "clear" {
			queue.clear()
		}
	}
}
func main() {
	ctx := context.Background()
	if os.Getenv("DATABASE_URL") == "" || os.Getenv("LINKED_BRIDGE_SECRET") == "" {
		log.Fatal("Missing configuration")
	}
	config, err := pgxpool.ParseConfig(os.Getenv("DATABASE_URL"))
	if err != nil {
		log.Fatal("Invalid database configuration")
	}
	config.MaxConns = 3
	config.MinConns = 0
	config.MaxConnIdleTime = 15 * time.Second
	config.HealthCheckPeriod = 5 * time.Second
	db, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()
	store.DeviceProps.Os = proto.String("SerendibAI")
	store.DeviceProps.PlatformType = waCompanionReg.DeviceProps_CHROME.Enum()
	m := &manager{db: db, sessions: map[string]*linkedSession{}}
	rows, err := db.Query(ctx, `select id::text from portal_whatsapp_links where enabled`)
	if err != nil {
		log.Fatal("Unable to load linked accounts")
	}
	for rows.Next() {
		var id string
		if rows.Scan(&id) == nil {
			m.start(id)
		}
	}
	rows.Close()
	http.HandleFunc("/connection", func(w http.ResponseWriter, r *http.Request) {
		if (r.Method != http.MethodPost && r.Method != http.MethodDelete) || subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), []byte("Bearer "+os.Getenv("LINKED_BRIDGE_SECRET"))) != 1 {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		var b struct {
			Customer string `json:"customer"`
			Enabled  bool   `json:"enabled"`
			Refresh  bool   `json:"refresh"`
		}
		if json.NewDecoder(http.MaxBytesReader(w, r.Body, 1024)).Decode(&b) != nil || !regexp.MustCompile(`^[0-9a-f-]{36}$`).MatchString(b.Customer) {
			http.Error(w, "Invalid request", 400)
			return
		}
		if r.Method == http.MethodDelete {
			if err := m.stop(b.Customer, true); err != nil {
				http.Error(w, "Unable to remove linked account", http.StatusInternalServerError)
				return
			}
		} else if b.Enabled {
			if b.Refresh {
				if err := m.restartPairing(b.Customer); err != nil {
					http.Error(w, "Unable to refresh pairing code", http.StatusConflict)
					return
				}
			} else {
				m.start(b.Customer)
			}
		} else {
			m.stop(b.Customer, false)
		}
		w.WriteHeader(http.StatusNoContent)
	})
	server := &http.Server{Addr: "127.0.0.1:8099", ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: 35 * time.Second, IdleTimeout: 30 * time.Second}
	log.Fatal(server.ListenAndServe())
}

func (m *manager) start(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.sessions[id]; !ok {
		ctx, cancel := context.WithCancel(context.Background())
		session := &linkedSession{cancel: cancel, done: make(chan struct{})}
		m.sessions[id] = session
		go func() {
			defer close(session.done)
			m.run(ctx, id)
		}()
	}
}

func (m *manager) restartPairing(id string) error {
	var status string
	if err := m.db.QueryRow(context.Background(), `select status from portal_whatsapp_links where id=$1`, id).Scan(&status); err != nil {
		return err
	}
	if status == "connected" {
		return errors.New("connection is already paired")
	}
	if err := m.stop(id, false); err != nil {
		return err
	}
	m.status(id, "connecting", "", "", 0)
	m.start(id)
	return nil
}

func (m *manager) stop(id string, remove bool) error {
	m.mu.Lock()
	session := m.sessions[id]
	var client *whatsmeow.Client
	if session != nil {
		client = session.client
	}
	m.mu.Unlock()
	if remove {
		if session != nil && client == nil {
			return errors.New("connection is starting")
		}
		if client != nil && client.Store.ID != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			err := client.Logout(ctx)
			cancel()
			if err != nil {
				return err
			}
		} else if session == nil {
			path := filepath.Join("sessions", id+".db")
			if _, err := os.Stat(path); err == nil {
				container, err := sqlstore.New(context.Background(), "sqlite", "file:"+path+"?_pragma=foreign_keys(1)&_pragma=busy_timeout(5000)", waLog.Noop)
				if err != nil {
					return err
				}
				device, err := container.GetFirstDevice(context.Background())
				container.Close()
				if err != nil {
					return err
				}
				if device.ID != nil {
					return errors.New("reconnect before unlinking")
				}
			} else if !os.IsNotExist(err) {
				return err
			}
		}
	}
	if session != nil {
		session.cancel()
		<-session.done
	}
	if remove {
		for _, suffix := range []string{"", "-wal", "-shm"} {
			if err := os.Remove(filepath.Join("sessions", id+".db"+suffix)); err != nil && !os.IsNotExist(err) {
				return err
			}
		}
	}
	return nil
}
