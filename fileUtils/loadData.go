package fileutils

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"
)

func ReadImages(filePath string) [][]int {
	f, err := os.OpenFile(filePath, os.O_RDONLY, os.ModeTemporary)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	// 1. Read the 4-byte Magic Number block into memory
	headerBytes := make([]byte, 4)
	if _, err := f.Read(headerBytes); err != nil {
		log.Fatal(err)
	}

	// 2. Read the dimension count directly from the 4th byte (Index 3)
	numDimensions := int(headerBytes[3])
	fmt.Printf("Detected Dimensions directly from byte 3: %d\n", numDimensions)

	// 3. Loop based on that dimension count to pull the size metadata
	dimensions := make([]int, numDimensions)
	for i := 0; i < numDimensions; i++ {
		if _, err := f.Read(headerBytes); err != nil {
			log.Fatal(err)
		}
		// Convert the 4 bytes to an int
		dimensions[i] = int(binary.BigEndian.Uint32(headerBytes))
		log.Println("Dimension", i, dimensions[i])
	}

	// 4. Extract sizes
	itemCount := dimensions[0]
	rows := dimensions[1]
	cols := dimensions[2]
	imageSize := rows * cols

	fmt.Printf("Item Count: %d\n", itemCount)
	fmt.Printf("Image Grid: %dx%d\n", rows, cols)
	fmt.Printf("Flat Size of single image: %d elements\n", imageSize)

	totalPixels := int(itemCount) * imageSize

	// 1. Allocate ONE giant flat pool for all pixel integers in the dataset
	// This is exactly 1 allocation in memory!
	allPixels := make([]int, totalPixels)

	// 2. Allocate your outer slice of images
	images := make([][]int, itemCount)

	// 3. Point each image row to its respective 784-element segment inside the giant pool
	for i := 0; i < int(itemCount); i++ {
		start := i * imageSize
		end := start + imageSize

		// Slice syntax doesn't copy data; it just creates a hyper-fast
		// pointer window into the main 'allPixels' block.
		images[i] = allPixels[start:end]
	}

	return images
}

func ReadLabels(filePath string) []int {
	f, err := os.OpenFile(filePath, os.O_RDONLY, os.ModeTemporary)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	int32Bytes := make([]byte, 4)

	_, err = f.Read(int32Bytes)
	if err != nil {
		log.Fatalf("error reading magic number %s", err.Error())

	}
	magicNumber := binary.BigEndian.Uint32(int32Bytes)
	fmt.Println("magic number: ", magicNumber) // Should be 2049

	_, err = f.Read(int32Bytes) // This now reads bytes 4, 5, 6, and 7 cleanly
	if err != nil {
		log.Fatalf("error reading number of items %s", err.Error())

	}
	itemCount := binary.BigEndian.Uint32(int32Bytes)
	fmt.Println("itemCount:    ", itemCount) // Should be 10000

	// Allocate a slice to hold all 10,000 labelsBytes
	labelsBytes := make([]byte, itemCount)

	// Read the entire rest of the file directly into your slice
	_, err = f.Read(labelsBytes)
	if err != nil {
		log.Fatal(err)
	}

	labelsInt := make([]int, len(labelsBytes))

	for i, labelByte := range labelsBytes {
		labelsInt[i] = int(labelByte)
	}

	return labelsInt
}
