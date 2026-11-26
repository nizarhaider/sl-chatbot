create table if not exists customers (
  id uuid primary key default gen_random_uuid(),
  business_name text not null,
  contact_name text not null
);

create table if not exists portal_agents (
  id uuid primary key default gen_random_uuid(),
  customer_id uuid not null references customers(id),
  name text not null,
  company_name text not null default '',
  greeting text not null default '',
  system_prompt text not null default '',
  languages text[] not null default array['English','Sinhala','Tamil'],
  max_calls integer not null default 3,
  phone_number_id text not null,
  status text not null default 'draft',
  updated_at timestamptz not null default now()
);

create table if not exists portal_calls (
  id text primary key,
  customer_id uuid not null references customers(id),
  agent_id uuid references portal_agents(id),
  customer_phone text,
  status text not null,
  transcript text,
  duration_seconds numeric,
  tokens integer,
  usage jsonb,
  events jsonb,
  recording_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists portal_agents_customer on portal_agents(customer_id);
create index if not exists portal_calls_customer_date on portal_calls(customer_id,created_at desc);
create table if not exists portal_text_usage (
  id text primary key,
  customer_id uuid not null references customers(id) on delete cascade,
  agent_id uuid references portal_agents(id) on delete set null,
  model text not null,
  usage jsonb not null,
  created_at timestamptz not null default now()
);
create index if not exists portal_text_usage_customer_date on portal_text_usage(customer_id,created_at desc);
create table if not exists portal_messages (
  id uuid primary key default gen_random_uuid(),
  customer_id uuid not null references customers(id) on delete cascade,
  agent_id uuid references portal_agents(id) on delete set null,
  connection_id uuid,
  contact_phone text not null,
  direction text not null check(direction in ('inbound','outbound')),
  body text not null,
  status text not null check(status in ('received','sent','failed')),
  source text not null check(source in ('linked_device','cloud_api')),
  provider_message_id text,
  reply_to_message_id text,
  model text,
  usage jsonb,
  created_at timestamptz not null default now()
);
create index if not exists portal_messages_contact_history on portal_messages(customer_id,contact_phone,created_at desc);
alter table portal_messages add column if not exists display_phone text;
create index if not exists portal_messages_customer_date on portal_messages(customer_id,created_at desc);
create unique index if not exists portal_messages_linked_provider on portal_messages(connection_id,provider_message_id) where connection_id is not null and provider_message_id is not null;
create unique index if not exists portal_messages_cloud_provider on portal_messages(customer_id,provider_message_id) where connection_id is null and provider_message_id is not null;
