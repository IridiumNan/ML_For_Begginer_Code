# App With Unix Socket

Unix Socket is a powerful local socket for inter-process communicate.

It makes multi-process exchange messages.

For the Tutorial on ML_For_Begginer. The author use `flask` framework.

But I'm familiar with golang backend.

So I write the go-python unix-socket micro-serves.

To run this demo. Ensure you are in the `3-Web-App` dir

This command will start the predict server written by python.

It will listen local unix socket

```sh
python ./unix_socket/predict_server.py
```

Then start the golang backend

It's zero dependency (just standard libs on golang)

```sh
go run ./unix_socket/main.go
```
