package paddle

import (
	"github.com/go-aie/xslices"
	"golang.org/x/exp/slices"
	"gonum.org/v1/gonum/mat"
)

type Matrix[E xslices.Number] struct {
	m *mat.Dense
}

func NewMatrixFromTensor[E xslices.Number](t Tensor) *Matrix[E] {
	tt := NewTypedTensor[E](t)
	if len(tt.Shape) != 2 {
		panic("t is not a matrix")
	}

	return NewMatrix(int(tt.Shape[0]), int(tt.Shape[1]), tt.Data)
}

func NewMatrix[E xslices.Number](r, c int, data []E) *Matrix[E] {
	return &Matrix[E]{
		m: mat.NewDense(r, c, xslices.NumberToFloat64(data)),
	}
}

func (m *Matrix[E]) Row(i int) []E {
	r := mat.Row(nil, i, m.m)
	return xslices.Float64ToNumber[E](r)
}

func (m *Matrix[E]) Col(j int) []E {
	c := mat.Col(nil, j, m.m)
	return xslices.Float64ToNumber[E](c)
}

func (m *Matrix[E]) Rows() [][]E {
	var rows [][]E

	r, _ := m.m.Dims()
	for i := 0; i < r; i++ {
		rows = append(rows, m.Row(i))
	}

	return rows
}

func (m *Matrix[E]) Cols() [][]E {
	var cols [][]E

	_, c := m.m.Dims()
	for j := 0; j < c; j++ {
		cols = append(cols, m.Col(j))
	}

	return cols
}

func (m *Matrix[E]) Dims() (r, c int) {
	return m.m.Dims()
}

func (m *Matrix[E]) Set(rowStart, rowEnd, colStart, colEnd int, value E) *Matrix[E] {
	return m.SetFunc(rowStart, rowEnd, colStart, colEnd, func(E) E {
		return value
	})
}

func (m *Matrix[E]) SetFunc(rowStart, rowEnd, colStart, colEnd int, f func(E) E) *Matrix[E] {
	for r := rowStart; r < rowEnd; r++ {
		for c := colStart; c < colEnd; c++ {
			v := m.m.At(r, c)
			newV := f(E(v))
			m.m.Set(r, c, float64(newV))
		}
	}
	return m
}

func (m *Matrix[E]) SetAll(value E) *Matrix[E] {
	return m.SetAllFunc(func(E) E {
		return value
	})
}

func (m *Matrix[E]) SetAllFunc(f func(E) E) *Matrix[E] {
	r, c := m.m.Dims()
	return m.SetFunc(0, r, 0, c, func(v E) E {
		return f(v)
	})
}

func (m *Matrix[E]) Norm() *Matrix[E] {
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

// Pad creates a new matrix, which expands the receiver m by r rows and c columns,
// and sets each expanded element to v.
func (m *Matrix[E]) Pad(r, c int, v E) *Matrix[E] {
	if r == 0 && c == 0 {
		return m
	}

	oldR, oldC := m.m.Dims()
	r += oldR
	c += oldC

	newM := NewMatrix[E](r, c, nil).SetAll(v)
	newM.m.Copy(m.m)
	return newM
}

func (m *Matrix[E]) RawData() []E {
	return xslices.Float64ToNumber[E](m.m.RawMatrix().Data)
}

// Equal implements the Equal method which will be used by go-cmp to determine equality.
func (m *Matrix[E]) Equal(other *Matrix[E]) bool {
	r, c := m.m.Dims()
	or, oc := other.Dims()
	if r != or || c != oc {
		return false
	}
	return slices.Equal(m.RawData(), other.RawData())
}
