package neuralObjects

import "fmt"

type NeuralLayer struct {
	nodes []Node
}

func (nl *NeuralLayer) GetNodes() []Node {
	return nl.nodes
}

func (nl *NeuralLayer) Update(prevLayer NeuralLayer) {
	for i := range nl.nodes {
		// fmt.Println("Prev layer node count", len(prevLayer.nodes))
		nl.nodes[i].Update(prevLayer.nodes)
	}
}

func NewNeuralLayer(nodeCount int, prevLayerNodeCount int) NeuralLayer {
	layer := NeuralLayer{}
	for i := 0; i < nodeCount; i++ {
		layer.nodes = append(layer.nodes, NewNode(prevLayerNodeCount))
	}
	return layer
}

func (layer *NeuralLayer) PrintNodeActivation() {
	vals := make([]float64, len(layer.nodes))
	for i, node := range layer.nodes {
		vals[i] = node.activation
	}
	fmt.Println("Layer Values are", vals)
}
