-- Football Predictor: онлайн-счётчик и история посещений.
-- Выполните этот файл целиком в Supabase -> SQL Editor -> New query -> Run.

create table if not exists public.site_sessions (
    session_id uuid primary key,
    first_seen timestamptz not null default now(),
    last_seen timestamptz not null default now(),
    first_page text not null default 'main',
    last_page text not null default 'main'
);

create index if not exists site_sessions_first_seen_idx
    on public.site_sessions (first_seen desc);

create index if not exists site_sessions_last_seen_idx
    on public.site_sessions (last_seen desc);

alter table public.site_sessions enable row level security;

-- Публичные ключи не получают прямого доступа к таблице.
revoke all on table public.site_sessions from anon, authenticated;

create or replace function public.register_site_session(
    p_session_id uuid,
    p_page text default 'main'
)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.site_sessions (
        session_id, first_seen, last_seen, first_page, last_page
    )
    values (
        p_session_id,
        now(),
        now(),
        left(coalesce(nullif(trim(p_page), ''), 'main'), 100),
        left(coalesce(nullif(trim(p_page), ''), 'main'), 100)
    )
    on conflict (session_id) do update
    set last_seen = now(),
        last_page = excluded.last_page;
end;
$$;

create or replace function public.heartbeat_site_session(
    p_session_id uuid,
    p_page text default 'main'
)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
    update public.site_sessions
    set last_seen = now(),
        last_page = left(coalesce(nullif(trim(p_page), ''), 'main'), 100)
    where session_id = p_session_id;

    if not found then
        perform public.register_site_session(p_session_id, p_page);
    end if;
end;
$$;

create or replace function public.get_site_public_summary()
returns table (
    online_now bigint,
    visits_today bigint,
    total_visits bigint
)
language sql
security definer
set search_path = public
as $$
    select
        count(*) filter (where last_seen >= now() - interval '2 minutes')::bigint,
        count(*) filter (where first_seen >= date_trunc('day', now()))::bigint,
        count(*)::bigint
    from public.site_sessions;
$$;

revoke all on function public.register_site_session(uuid, text) from public;
revoke all on function public.heartbeat_site_session(uuid, text) from public;
revoke all on function public.get_site_public_summary() from public;

grant execute on function public.register_site_session(uuid, text) to anon, authenticated;
grant execute on function public.heartbeat_site_session(uuid, text) to anon, authenticated;
grant execute on function public.get_site_public_summary() to anon, authenticated;
