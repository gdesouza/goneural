package neuron

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSigmoid(t *testing.T) {
	x := 0.0
	expected := 0.5
	result := sigmoid(x)
	assert.Equal(t, expected, result)

	x = 4.0
	expected = 1.0
	result = sigmoid(x)
	assert.Equal(t, true, expected-result < 0.02)
}

func TestSigmoidPrime(t *testing.T) {
	x := 0.0
	expected := 0.25
	result := sigmoidPrime(x)
	assert.Equal(t, expected, result)

	x = 4.0
	expected = 0
	result = sigmoidPrime(x)
	assert.Equal(t, true, expected-result < 0.02)
}

func TestDendrite(t *testing.T) {
	d := dendrite{
		Value:  1.0,
		Weight: 1.0,
	}
	expected := 1.0
	result := d.GetOutput()
	assert.Equal(t, expected, result)
}

func TestAxon(t *testing.T) {
	values := []float64{1.0, 2.0}
	weights := []float64{1.0, 2.0}
	a := NewAxon(
		&dendrite{values[0], weights[0]},
		&dendrite{values[1], weights[1]},
	)

	// pre-conditions
	for i := 0; i < len(a.dendrites); i++ {
		expected := values[i] * weights[i]
		result := a.dendrites[i].GetOutput()
		assert.Equal(t, expected, result)
	}

	signal := 5.0
	a.Propagate(signal)

	// post-conditions
	for i := 0; i < len(a.dendrites); i++ {
		expected := signal * weights[i]
		result := a.dendrites[i].GetOutput()
		assert.Equal(t, expected, result)
	}
}

func TestNeuron(t *testing.T) {
	d1 := &dendrite{2.0, 1.0}
	d2 := &dendrite{0.0, 5.0}

	network := make([]*neuron, 2)
	network[1] = NewNeuron(NewAxon(&dendrite{}), d2)
	network[0] = NewNeuron(NewAxon(d2), d1)

	for _, neuron := range network {
		neuron.Process()
	}

	assert.Equal(t, sigmoid(d1.GetOutput()), network[0].signal)
	assert.Equal(t, network[0].signal, network[1].dendrites[0].Value)
	assert.Equal(t, sigmoid(d2.GetOutput()), network[1].signal)
}
