package main

import (
	"fmt"
	"log"

	fileutils "github.com/shreyghildiyal/goNeuralNet/fileUtils"
	"github.com/shreyghildiyal/goNeuralNet/neuralObjects"
	"github.com/shreyghildiyal/goNeuralNet/utils"
	// "github.com/shreyghildiyal/goNeuralNet/neuralObjects"
)

func main() {
	fmt.Println("Hello World")

	// a := rand.Float64()
	// b := rand.Float64()
	// DoNetworkStuff()

	labels, images := DoFileReadingStuff()

	labelArr := utils.NormalizeLabels(labels, 10) // converts the labels into an array of ten float64 values. all except one are set to zero.
	_ = labelArr
	imageNormalizer := utils.NewImageNormalizer(0, 255, 0, 1)
	normalizedImages := imageNormalizer.NormalizeImages(images) // sets the pixel value between 0 and 1

	log.Println("Number of labels", len(labels))
	log.Println("Number of images", len(images))

	network := neuralObjects.NewNetwork([]int{len(normalizedImages[0]), 5, 5, 5, len(labelArr[0])})

	network.Train(normalizedImages, labelArr, 10, 20)

	// break the training data
}

func DoFileReadingStuff() ([]int, [][]int) {
	labels := fileutils.ReadLabels("datasets/t10k-labels.idx1-ubyte")
	images := fileutils.ReadImages("datasets/t10k-images.idx3-ubyte")

	return labels, images
}

// func DoNetworkStuff() {
// 	network := neuralObjects.NewNetwork(2, []int{2, 2}, 2)

// 	for i := 0; i < 5; i++ {

// 		a := float64(i)
// 		b := float64(i + 1)

// 		// network.Update([]float64{a, b})

// 		// network.PrintAllLayerActivations()
// 	}
// }
