package paddle

import (
	"gonum.org/v1/gonum/mat"
)

type Matrix struct {
	m *mat.Dense
}

func NewMatrix[E Number](t TypedTensor[E]) *Matrix {
	if len(t.Shape) != 2 {
		panic("t is not a matrix")
	}

	m := mat.NewDense(int(t.Shape[0]), int(t.Shape[1]), NumberToFloat64(t.Data))
	return &Matrix{m: m}
}

func (m *Matrix) Norm() *Matrix {
	r, c := m.m.Dims()

	norm := m.m.Norm(2) // the square root of the sum of the squares of the elements
	data := make([]float64, r*c)
	for i := 0; i < len(data); i++ {
		data[i] = norm
	}

	normDense := mat.NewDense(r, c, data)
	m.m.DivElem(m.m, normDense)

	return m
}

func (m *Matrix) RawData() []float64 {
	return m.m.RawMatrix().Data
}

func Row[E Number](m *Matrix, i int) []E {
	r := mat.Row(nil, i, m.m)
	return Float64ToNumber[E](r)
}

func Col[E Number](m *Matrix, j int) []E {
	c := mat.Col(nil, j, m.m)
	return Float64ToNumber[E](c)
}

func Rows[E Number](m *Matrix) [][]E {
	var rows [][]E

	r, _ := m.m.Dims()
	for i := 0; i < r; i++ {
		rows = append(rows, Row[E](m, i))
	}

	return rows
}

func Cols[E Number](m *Matrix) [][]E {
	var cols [][]E

	_, c := m.m.Dims()
	for j := 0; j < c; j++ {
		cols = append(cols, Col[E](m, j))
	}

	return cols
}
