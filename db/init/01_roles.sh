#!/bin/bash
# 컨테이너 최초 기동 시 1회 실행된다 (docker-entrypoint-initdb.d, superuser 권한).
#
# 계정 2개를 만든다 (spec "저장" 절 — 애플리케이션 계정과 text-to-SQL 실행용 read-only 계정 분리):
#   * 앱 계정      — public 스키마에 객체를 만들고 읽고 쓴다.
#   * read-only 계정 — NOLOGIN 그룹 `reply_gate_readers` 의 멤버로서 SELECT 만 상속받는다.
#     테이블 단위 SELECT 부여는 `db/schema.sql` 이 그 그룹에 한다(계정 이름이 환경 변수라
#     정적 DDL 에 적을 수 없기 때문 — 그룹 이름은 비밀이 아니므로 고정한다).
#
# 비밀은 환경 변수에서만 온다. 이 파일에는 값이 없다. psql 의 `:'var'` 는 값을 SQL 문자열
# 리터럴로 안전하게 인용하므로 비밀번호에 특수문자가 있어도 깨지지 않는다.
set -euo pipefail

: "${POSTGRES_DB:?POSTGRES_DB 가 필요하다}"
: "${POSTGRES_APP_USER:?POSTGRES_APP_USER 가 필요하다}"
: "${POSTGRES_APP_PASSWORD:?POSTGRES_APP_PASSWORD 가 필요하다 (.env 에 채운다)}"
: "${POSTGRES_RO_USER:?POSTGRES_RO_USER 가 필요하다}"
: "${POSTGRES_RO_PASSWORD:?POSTGRES_RO_PASSWORD 가 필요하다 (.env 에 채운다)}"

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" \
     --dbname "$POSTGRES_DB" \
     --set=db_name="$POSTGRES_DB" \
     --set=app_user="$POSTGRES_APP_USER" \
     --set=app_password="$POSTGRES_APP_PASSWORD" \
     --set=ro_user="$POSTGRES_RO_USER" \
     --set=ro_password="$POSTGRES_RO_PASSWORD" <<'SQL'

-- read-only 그룹 (로그인 불가, 비밀 아님 → 이름 고정).
SELECT 'CREATE ROLE reply_gate_readers NOLOGIN'
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'reply_gate_readers')
\gexec

-- 애플리케이션 계정 (읽기/쓰기 + 스키마 생성).
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'app_user', :'app_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'app_user')
\gexec

-- text-to-SQL 실행 전용 계정 (SELECT 만 — 안전장치 1).
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'ro_user', :'ro_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'ro_user')
\gexec

SELECT format('GRANT reply_gate_readers TO %I', :'ro_user')
\gexec

-- DB 접속: PUBLIC 에서 회수한다. TEMP 테이블 생성 권한(PUBLIC 기본값)도 함께 사라지므로
-- read-only 계정은 임시 테이블조차 만들 수 없다.
REVOKE ALL ON DATABASE :"db_name" FROM PUBLIC;
SELECT format('GRANT CONNECT ON DATABASE %I TO %I', :'db_name', :'app_user')
\gexec
GRANT CONNECT ON DATABASE :"db_name" TO reply_gate_readers;

-- public 스키마: 앱 계정만 객체를 만든다. read-only 는 참조(USAGE)만 한다.
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO reply_gate_readers;
SELECT format('GRANT USAGE, CREATE ON SCHEMA public TO %I', :'app_user')
\gexec

SQL
