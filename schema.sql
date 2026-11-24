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
