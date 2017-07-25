// Copyright 2017 The BigVector Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"unicode"
)

const (
	dataLocation = "data/"
	vectorSize   = 1024
	bufferSize   = 17
	queryBook    = "pg1661.txt"
	queryWord    = "sun"
)

var authors = map[string]string{
	"data/pg1661.txt":  "Arthur Conan Doyle",
	"data/pg174.txt":   "Oscar Wilde",
	"data/pg1342.txt":  "Jane Austen",
	"data/pg98.txt":    "Charles Dickens",
	"data/pg2852.txt":  "A. Conan Doyle",
	"data/pg74.txt":    "Mark Twain (Samuel Clemens)",
	"data/pg76.txt":    "Mark Twain (Samuel Clemens)",
	"data/pg2591.txt":  "The Brothers Grimm",
	"data/pg27827.txt": "Vatsyayana",
	"data/pg1232.txt":  "Nicolo Machiavelli",
	"data/pg23.txt":    "Frederick Douglass",
	"data/pg16328.txt": "Lesslie Hall",
	"data/pg11.txt":    "Lewis Carroll",
	"data/pg2265.txt":  "William Shakespeare",
	"data/pg5200.txt":  "Franz Kafka",
	"data/pg2267.txt":  "William Shakespeare",
	"data/pg1112.txt":  "William Shakespeare",
	"data/pg1952.txt":  "Charlotte Perkins Gilman",
	"data/pg2701.txt":  "Herman Melville",
	"data/pg6130.txt":  "Homer",
	"data/pg4300.txt":  "James Joyce",
	"data/244-0.txt":   "Arthur Conan Doyle",
	"data/2097.txt":    "Arthur Conan Doyle",
}

type Buffer struct {
	Buffer []string
	Index  int
}

func NewBuffer() *Buffer {
	return &Buffer{
		Buffer: make([]string, bufferSize),
	}
}

func (b *Buffer) Push(a string) {
	b.Buffer[b.Index] = a
	b.Index = (b.Index + 1) % bufferSize
}

type BigVector struct {
	Vector []int64
	Words  map[string][]int64
	Name   string
}

func NewBigVector(size int) *BigVector {
	return &BigVector{
		Vector: make([]int64, size),
		Words:  make(map[string][]int64),
	}
}

func hash(a string) uint64 {
	h := fnv.New64()
	h.Write([]byte(a))
	return h.Sum64()
}

func (b *BigVector) ProcessFile(name string) {
	file, err := os.Open(name)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	vector, cache, reader, word, buffer, size :=
		b.Vector, make(map[uint64][]int8), bufio.NewReader(file), "", NewBuffer(), len(b.Vector)
	lookup := func(a string) []int8 {
		h := hash(a)
		transform, found := cache[h]
		if found {
			return transform
		}
		transform = make([]int8, size)
		rnd := rand.New(rand.NewSource(int64(h)))
		for i := range vector {
			// https://en.wikipedia.org/wiki/Random_projection#More_computationally_efficient_random_projections
			// make below distribution function of vector element index
			switch rnd.Intn(6) {
			case 0:
				transform[i] = 1
			case 1:
				transform[i] = -1
			}
		}
		cache[h] = transform
		return transform
	}
	for {
		r, _, err := reader.ReadRune()
		if err != nil {
			break
		}
		if unicode.IsLetter(r) || r == '\'' {
			word += string(unicode.ToLower(r))
		} else if word != "" {
			transform := lookup(buffer.Buffer[(buffer.Index+bufferSize-1)%bufferSize] + word)
			for i, t := range transform {
				vector[i] += int64(t)
			}

			center := buffer.Buffer[(buffer.Index+bufferSize/2)%bufferSize]
			wordVector := b.Words[center]
			if wordVector == nil {
				wordVector = make([]int64, size)
				b.Words[center] = wordVector
			}

			/*for i := 1; i < bufferSize; i++ {
				current := buffer.Buffer[(buffer.Index+i)%bufferSize]
				if current == center {
					continue
				}
				transform := lookup(current)
				for i, t := range transform {
					wordVector[i] += int64(t)
				}
			}*/

			last := buffer.Buffer[buffer.Index]
			for i := 1; i < bufferSize; i++ {
				current := buffer.Buffer[(buffer.Index+i)%bufferSize]
				if current == center {
					continue
				}
				transform := lookup(last + current)
				for i, t := range transform {
					wordVector[i] += int64(t)
				}
				last = current
			}

			buffer.Push(word)
			word = ""
		}
	}
	b.Name = name
}

func (b *BigVector) Distance(a *BigVector) float64 {
	/*var d int64
		for i, j := range b.Vector {
			diff := j - a.Vector[i]
			d += diff * diff
		}
	  return float64(d)*/
	return Similarity(a.Vector, b.Vector)
}

func Similarity(a, b []int64) float64 {
	dot, aa, bb := 0.0, 0.0, 0.0
	for i, j := range b {
		x, y := float64(a[i]), float64(j)
		dot += x * y
		aa += x * x
		bb += y * y
	}
	return dot / math.Sqrt(aa*bb)
}

type Distance struct {
	D    float64
	Name string
}

type Distances []Distance

func (d Distances) Len() int {
	return len(d)
}

func (d Distances) Swap(i, j int) {
	d[i], d[j] = d[j], d[i]
}

func (d Distances) Less(i, j int) bool {
	return d[i].D > d[j].D
}

func main() {
	data, err := os.Open(dataLocation)
	if err != nil {
		panic(err)
	}
	defer data.Close()

	files, err := data.Readdir(-1)
	if err != nil {
		panic(err)
	}
	vectors, wait := make([]*BigVector, len(files)), sync.WaitGroup{}
	wait.Add(len(files))
	process := func(name string, vector *BigVector) {
		vector.ProcessFile(name)
		fmt.Println(name)
		wait.Done()
	}
	var query *BigVector
	for i, file := range files {
		fileName := dataLocation + file.Name()
		vectors[i] = NewBigVector(vectorSize)
		if file.Name() == queryBook {
			query = vectors[i]
		}
		go process(fileName, vectors[i])
	}
	wait.Wait()

	words := make(map[string][]int64)
	for i := range vectors {
		for word, vector := range vectors[i].Words {
			wordVector := words[word]
			if wordVector == nil {
				wordVector = make([]int64, vectorSize)
				words[word] = wordVector
			}
			for j, element := range vector {
				wordVector[j] += element
			}
		}
	}

	fmt.Println("\nresults:")
	distances := make(Distances, len(files))
	for i := range distances {
		distances[i].D = query.Distance(vectors[i])
		distances[i].Name = vectors[i].Name
	}
	sort.Sort(distances)
	for d := range distances {
		fmt.Printf("%v, %v\n", authors[distances[d].Name], distances[d].Name)
	}

	best := [20]struct {
		best float64
		word string
	}{}
	insert := func(b float64, l string) {
		c := 0
		for c < len(best) && b < best[c].best {
			c++
		}
		for c < len(best) {
			b, best[c].best, l, best[c].word = best[c].best, b, best[c].word, l
			c++
		}
	}

	queryVector := words[queryWord]
	for word, vector := range words {
		insert(Similarity(queryVector, vector), word)
	}
	fmt.Printf("\nword match:\n")
	for b := range best {
		fmt.Println(best[b].word)
	}
}
