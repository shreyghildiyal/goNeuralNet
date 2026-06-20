package main

import (
	"fmt"
	"log"

	fileutils "github.com/shreyghildiyal/goNeuralNet/fileUtils"
	"github.com/shreyghildiyal/goNeuralNet/neuralObjects"
)

func main() {
	fmt.Println("Hello World")

	// a := rand.Float64()
	// b := rand.Float64()
	// DoNetworkStuff()

	labels, images := DoFileReadingStuff()

	log.Println("Number of labels", len(labels))
	log.Println("Number of images", len(images))

	// fmt.Println("Input Layer")
	// network.GetInputLayer().PrintCurrentNodeValues()
	// fmt.Println("Output Layer")
	// network.GetOuputLayer().PrintCurrentNodeValues()
}

func DoFileReadingStuff() ([]int, [][]int) {
	labels := fileutils.ReadLabels("datasets/t10k-labels.idx1-ubyte")
	images := fileutils.ReadImages("datasets/t10k-images.idx3-ubyte")

	return labels, images
}

func DoNetworkStuff() {
	network := neuralObjects.NewNetwork(2, []int{2, 2}, 2)

	for i := 0; i < 5; i++ {

		a := float64(i)
		b := float64(i + 1)

		network.Update([]float64{a, b})

		network.PrintAllLayerActivations()
	}
}
