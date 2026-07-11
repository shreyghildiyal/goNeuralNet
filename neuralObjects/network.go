package neuralObjects

import (
	"fmt"

	"github.com/shreyghildiyal/goNeuralNet/activation"
)

type Network struct {
	layers       []NeuralLayer
	learningRate float64
}

func NewNetwork(layerCounts []int, learningRate float64) Network {
	network := Network{}
	// network.layers = NewNeuralLayer(inputLayerNodeCount, 0, activation.FastSigmoid)

	// prevNodeCount := inputLayerNodeCount
	prevNodeCount := 1

	for _, layerNodeCount := range layerCounts {
		network.layers = append(network.layers, NewNeuralLayer(layerNodeCount, prevNodeCount, activation.FastSigmoid))
		prevNodeCount = layerNodeCount
	}
	network.learningRate = learningRate

	// network.outputLayer = NewNeuralLayer(outputLayerNodeCount, prevNodeCount, activation.FastSigmoid)
	return network
}

func (network *Network) GetOuputLayer() *NeuralLayer {
	return &network.layers[len(network.layers)-1]
}

func (network *Network) GetInputLayer() *NeuralLayer {
	return &network.layers[0]
}

func (network *Network) GetLayer(i int) *NeuralLayer {
	return &network.layers[i]
}

func (network *Network) Train(images [][]float64, labels [][]float64, epochs, batchSize int) {

	for i := 0; i < 10; i++ {
		fmt.Println("LabelArr", i, labels[i])
	}

	for i := 0; i < epochs; i++ {
		for j := 0; j < len(images); j += batchSize {
			fmt.Printf("Runningbatch with params epoch: %d | j: %d | batchSize: %d\n ", i, j, batchSize)
			network.runBatch(images, labels, j, batchSize)
		}

	}
	// one batch runs through upto a certain number of input values before updating the network values

}

func (network *Network) runBatch(images [][]float64, labels [][]float64, startIndex, batchsize int) {

	// create an array to hold the total errors we see
	// errArr := make([]float64, len(network.layers[len(network.layers)-1].weights))

	for i := 0; i < batchsize && startIndex+i < len(images); i++ {
		network.forwardPass(images[startIndex+i])
		network.backPropagate(labels[startIndex+i])
		// if i%100 == 0

		// if startIndex+i == 101 { // Just print for the first image of the batch
		fmt.Printf("Outputs: v%0.3f | Target: v%0.3f | Output Deltas: v%0.6f | i: %d | startIndex: %d\n",
			network.layers[len(network.layers)-1].lastOutputs, labels[startIndex+i], network.layers[len(network.layers)-1].currentBiasErrors, i, startIndex)
		// }

	}

	for i := 1; i < len(network.layers); i++ {
		network.layers[i].ApplyAndResetGradients(float64(batchsize), network.learningRate)
	}
}

func (network *Network) backPropagate(targetLabel []float64) {

	errSlice := make([]float64, len(targetLabel))

	layersCount := len(network.layers)

	for i := 0; i < len(targetLabel); i++ {
		errSlice[i] = targetLabel[i] - network.layers[layersCount-1].lastOutputs[i]
	}

	for i := len(network.layers) - 1; i > 0; i-- {
		errSlice = network.layers[i].BackPropagate(errSlice, &network.layers[i-1])
	}

	// panic("unimplemeted")
}

func (network *Network) forwardPass(input []float64) {

	// set the values as the output of the input layer
	for i := range network.layers[0].lastOutputs {
		network.layers[0].lastOutputs[i] = input[i]
	}

	for i := 1; i < len(network.layers); i++ {
		network.layers[i].ForwardUpdate(&network.layers[i-1])
	}
}

func (network *Network) GetResult(input []float64) []float64 {

	network.forwardPass(input)

	result := make([]float64, len(network.layers[len(network.layers)-1].lastOutputs))
	copy(result, network.layers[len(network.layers)-1].lastOutputs)
	return result
}
