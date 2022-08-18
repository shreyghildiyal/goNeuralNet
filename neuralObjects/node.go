package neuralObjects

import (
	"math/rand"

	"github.com/shreyghildiyal/goNeuralNet/utils"
)

type Node struct {
	weights    []float64
	bias       float64
	activation float64
}

func (n *Node) Update(prevNodes []Node) {
	activationValue := 0.0
	for i, prevNode := range prevNodes {
		activationValue += prevNode.activation * n.weights[i]

		// fmt.Printf("weight %f, prevNodeCurrent %f\n", n.weights[i], prevNode.currentValue)
	}
	// fmt.Println("current Value", currentValue)
	n.activation = utils.RelU(activationValue + n.bias)
}

func NewNode(prevLayerNodeCount int) Node {
	node := Node{}
	node.activation = 0
	for i := 0; i < prevLayerNodeCount; i++ {
		node.weights = append(node.weights, rand.Float64())
	}
	return node
}

func (node *Node) GetWeigths() []float64 {
	return node.weights
}

func (node *Node) GetActivation() float64 {
	return node.activation
}
