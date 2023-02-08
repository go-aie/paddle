package paddle

import (
	"golang.org/x/exp/constraints"
)

type Number interface {
	constraints.Integer | constraints.Float
}

func NumberToInt64[E Number](v []E) []int64 {
	var result []int64
	for _, vv := range v {
		result = append(result, int64(vv))
	}
	return result
}

func NumberToFloat64[E Number](v []E) []float64 {
	var result []float64
	for _, vv := range v {
		result = append(result, float64(vv))
	}
	return result
}

func Float64ToNumber[E Number](v []float64) []E {
	var result []E
	for _, vv := range v {
		result = append(result, E(vv))
	}
	return result
}
