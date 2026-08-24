-- One-time PostgreSQL bridge for the frozen FlavourBench 0001 database.
--
-- This is the JSON-safe definition introduced by migration 0016.  Keeping the
-- bridge outside the Alembic history lets us repair the live 0012 function
-- without rewriting an already-published migration.
CREATE OR REPLACE FUNCTION public.flavourbench_0010_execution_contract_immutable()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE
    old_json jsonb := pg_catalog.to_jsonb(OLD);
    new_json jsonb := pg_catalog.to_jsonb(NEW);
BEGIN
    IF TG_TABLE_NAME = 'season_models'
       AND old_json ->> 'manifest_sha256' NOT IN ('', 'unfrozen', 'unresolved')
       AND (
           old_json ->> 'execution_backend' IS DISTINCT FROM
               new_json ->> 'execution_backend'
           OR old_json -> 'backend_contract_json' IS DISTINCT FROM
               new_json -> 'backend_contract_json'
           OR old_json ->> 'backend_contract_sha256' IS DISTINCT FROM
               new_json ->> 'backend_contract_sha256'
       ) THEN
        RAISE EXCEPTION 'frozen season execution backend is immutable';
    ELSIF TG_TABLE_NAME = 'battles'
       AND old_json -> 'provider_reservations_json' IS DISTINCT FROM
           new_json -> 'provider_reservations_json' THEN
        RAISE EXCEPTION 'battle provider reservation contract is immutable';
    ELSIF TG_TABLE_NAME = 'response_arms'
       AND old_json ->> 'execution_backend' IS DISTINCT FROM
           new_json ->> 'execution_backend' THEN
        RAISE EXCEPTION 'response-arm execution backend is immutable';
    END IF;
    RETURN NEW;
END;
$$;
