package main

import (
	"fmt"
	"math/rand"

	"github.com/shreyghildiyal/goNeuralNet/neuralObjects"
)

func main() {
	fmt.Println("Hello World")

	network := neuralObjects.NewNetwork(2, []int{2, 2}, 1)

	for i := 0; i < 3; i++ {
		a := rand.Float64()
		b := rand.Float64()

		network.Update([]float64{a, b})

		fmt.Println(a, b, network.GetOuputLayer().GetNodes()[0].GetActivation())
	}

	// fmt.Println("Input Layer")
	// network.GetInputLayer().PrintCurrentNodeValues()
	// fmt.Println("Output Layer")
	// network.GetOuputLayer().PrintCurrentNodeValues()
}
