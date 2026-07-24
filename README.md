# word-agent — словарик для Альбины

Сайт: https://fosttt.github.io/word-agent/

Вводишь английское слово или выражение — получаешь разбор по шаблону:
эквивалент, уровень (A1–C2), частотность, стиль, актуальность, словосочетания
с примерами, семейство слов, этимология.

## Устройство
- `index.html` — статичный сайт (GitHub Pages).
- `server/` — FastAPI-прокси над `claude -p --model haiku` (подписка Claude Code),
  крутится на workspace-VPS как systemd-сервис `word-agent`, порт 443,
  HTTPS через Let's Encrypt для `198-46-253-50.sslip.io` (автопродление certbot,
  deploy-hook рестартит сервис).
- Кэш ответов на диске (`~/.cache/word_agent`), один запрос одновременно,
  rate limit 10/мин с IP.
