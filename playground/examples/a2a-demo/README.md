```
export DEEPSEEK_API_KEY=

cargo run

curl 'http://localhost:9999/jsonrpc' \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "SendMessage",
    "params": {
      "message": {
        "messageId": "1",
        "role": "ROLE_USER",
        "parts": [
          {
            "text": "帮我生成 10 个随机数"
          }
        ]
      }
    }
  }'
```
