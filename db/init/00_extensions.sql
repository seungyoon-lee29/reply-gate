-- 컨테이너 최초 기동 시 1회 실행된다 (docker-entrypoint-initdb.d, superuser 권한).
-- pgvector 확장은 superuser 만 설치할 수 있으므로 여기서 깐다 — 이후 스키마 DDL 은
-- 애플리케이션 계정이 적용한다.
CREATE EXTENSION IF NOT EXISTS vector;
