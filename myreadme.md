## local IP in ipv4 LAN
```
192.168.14.122
```

## create certificate
```
mkcert -install && mkcert -cert-file cert.pem -key-file key.pem 192.168.14.122 localhost 127.0.0.1
```

## In apple Safari
```
https://192.168.14.122:8012?ws=wss://192.168.8.102:8012
```