package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"net"
)

const socketPath = "/tmp/py-predict.sock"

type Request struct {
	Latitude  float32 `json:"Latitude"`
	Longitude float32 `json:"Longitude"`
	Seconds   float32 `json:"Seconds"`
}

func main() {
	c, err := net.Dial("unix", socketPath)
	if err != nil {
		log.Fatalf("error while establish new connection: %s", err)
	}

	defer c.Close()
	// 20.0      gb   -2.916667  53.200000

	req := Request{
		Seconds:   20,
		Longitude: -2.917,
		Latitude:  53.2,
	}

	data, _ := json.Marshal(req)

	length := len(data)
	ulength := uint32(length)

	blength := make([]byte, 4)
	binary.BigEndian.PutUint32(blength, ulength)

	fmt.Printf("the length of blength: %d, the content of blength: %v\n", len(blength), blength)
	_, err = c.Write(blength)
	if err != nil {
		log.Fatalf("error while writing length: %s", err)
	}

	_, err = c.Write(data)
	if err != nil {
		log.Fatalf("error while writing data: %s", err)
	}
}
