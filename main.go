package main

import (
	"fmt"

	"github.com/shreyghildiyal/goNeuralNet/neuralObjects"
)

func main() {
	fmt.Println("Hello World")

	network := neuralObjects.NewNetwork(2, []int{2, 2}, 2)

	for i := 0; i < 5; i++ {
		// a := rand.Float64()
		// b := rand.Float64()

		a := float64(i)
		b := float64(i + 1)

		network.Update([]float64{a, b})

		network.PrintAllLayerActivations()
	}

	// fmt.Println("Input Layer")
	// network.GetInputLayer().PrintCurrentNodeValues()
	// fmt.Println("Output Layer")
	// network.GetOuputLayer().PrintCurrentNodeValues()
}
