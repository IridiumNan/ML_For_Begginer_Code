package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"net"
)

// this golang demo file just create a new connection then send a test request with features
// in practice, you should reuse the unix socket connection instead of creating it for each task

const socketPath = "/tmp/py-predict.sock"

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

	// write the length of send data
	// so that server and read properly for this loop
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

	fmt.Println("result Country: ", res.Country)
}
