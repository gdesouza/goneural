package neuron

import "math"

// Activation functions

// Sigmoid function
func sigmoid(x float64) float64 {
	return 1.0 / (1 + math.Exp(-x))
}

// derivative of the sigmoid function
func sigmoidPrime(x float64) float64 {
	sig := sigmoid(x)
	return sig * (1.0 - sig)
}

// Dendrites hold weighted values
type dendrite struct {
	Value  float64
	Weight float64
}

func (d *dendrite) GetOutput() float64 {
	return d.Value * d.Weight
}

// an axon contains connection to other neurons' dendrites
type axon struct {
	dendrites []*dendrite
}

func (a *axon) Propagate(signal float64) {
	for i := 0; i < len(a.dendrites); i++ {
		a.dendrites[i].Value = signal
	}
}

func NewAxon(dendrites ...*dendrite) axon {
	return axon{dendrites}
}

// The basic functions of a neuron
// - Receive signals (or information).
// - Integrate incoming signals (to determine whether or not the information
//   should be passed along).
// - Communicate signals to target cells (other neurons or muscles or glands).
type neuron struct {
	// dendrites hold input values from other neurons
	dendrites []*dendrite

	// the neuron receives the inputs and calculates the output signal
	signal float64

	// the axon connects the neuron with other neurons' dentrites
	axon axon
}

func (n *neuron) activate(signal float64) float64 {
	return sigmoid(signal)
}

func (n *neuron) Process() {
	signal := 0.0
	for _, d := range n.dendrites {
		signal += d.GetOutput()
	}
	n.signal = n.activate(signal)

	n.axon.Propagate(n.signal)
}

func NewNeuron(a axon, ds ...*dendrite) *neuron {
	return &neuron{
		dendrites: ds,
		axon:      a,
	}
}
