package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"
	"strconv"
)

// WARN: Run this server on 3-Web-App dir or change this dir variable
const staticDir = "unix_socket/static"

// chnage this addr for expose
// e.g. 0.0.0.0:8081
const httpAddr = "127.0.0.1:8081"

type Request struct {
	// NOTE: All features should be defined with sequance alphabet
	// so that the python predict engine can parse it properly
	Latitude  float32 `json:"Latitude"`
	Longitude float32 `json:"Longitude"`
	Seconds   float32 `json:"Seconds"`
}

type Respond struct {
	Country string `json:"Country"`
}

const socketPath = "/tmp/py-predict.sock"

func newConn() net.Conn {
	c, err := net.Dial("unix", socketPath)
	if err != nil {
		log.Fatalf("error while establishing new connection: %s", err)
	}

	return c
}

func Router() http.Handler {
	r := http.NewServeMux()

	r.HandleFunc("/predict", predictHandler)

	r.Handle("/", http.FileServerFS(os.DirFS(staticDir)))

	return r
}

func predictHandler(w http.ResponseWriter, r *http.Request) {
	latitudeStr := r.FormValue("latitude")
	latitude, err := strconv.ParseFloat(latitudeStr, 32)
	if err != nil {
		slog.Error("error while parsing latitude", "err", err, "str", latitudeStr)
	}

	longitudeStr := r.FormValue("longitude")
	longitude, err := strconv.ParseFloat(longitudeStr, 32)
	if err != nil {
		slog.Error("error while parsing longitude", "err", err, "str", longitudeStr)
	}

	secondStr := r.FormValue("seconds")
	seconds, err := strconv.ParseFloat(secondStr, 32)
	if err != nil {
		slog.Error("error while parsing seconds", "err", err, "str", secondStr)
	}

	reqData := Request{
		Latitude:  float32(latitude),
		Longitude: float32(longitude),
		Seconds:   float32(seconds),
	}

	respData := predict(reqData, newConn())

	fmt.Fprintf(w, "the country predicted is %s", respData.Country)
}

func predict(reqData Request, c io.ReadWriter) (resp Respond) {
	data, _ := json.Marshal(reqData)

	// write the length of send data
	// so that server and read properly for this loop
	length := len(data)
	ulength := uint32(length)
	blength := make([]byte, 4)
	binary.BigEndian.PutUint32(blength, ulength)

	_, err := c.Write(blength)
	if err != nil {
		log.Fatalf("error while writing length: %s", err)
	}

	_, err = c.Write(data)
	if err != nil {
		log.Fatalf("error while writing data: %s", err)
	}

	// read the response data length first
	// then create a new []byte for receiving data
	c.Read(blength)
	respLen := binary.BigEndian.Uint32(blength)
	respByte := make([]byte, respLen)
	_, err = c.Read(respByte)
	if err != nil {
		log.Fatalf("error while reading for resp body, err:%s", err)
	}

	var res Respond
	err = json.Unmarshal(respByte, &res)
	if err != nil {
		log.Fatalf("error while unmarshal data: %s", err)
	}
	return res
}

func main() {
	r := Router()

	err := http.ListenAndServe(httpAddr, r)
	if err != nil {
		log.Fatal("error while starting server", err)
	}
}
