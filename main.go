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

// CircularBuffer is a circular buffer of size bufferSize
type CircularBuffer struct {
	Buffer          []string
	Index, Previous int
}

// NewCircularBuffer creates a new circular buffer of size bufferSize
func NewCircularBuffer() *CircularBuffer {
	return &CircularBuffer{
		Buffer: make([]string, bufferSize),
	}
}

// Push adds a new string to the end of the buffer
func (c *CircularBuffer) Push(a string) {
	c.Buffer[c.Index] = a
	c.Index, c.Previous = (c.Index+1)%bufferSize, c.Index
}

// Item returns the string at index relative to the beginning of the buffer
func (c *CircularBuffer) Item(index int) string {
	return c.Buffer[(c.Index+index)%bufferSize]
}

// GetPrevious gets the string just inserted into the buffer
func (c *CircularBuffer) GetPrevious() string {
	return c.Buffer[c.Previous]
}

// BigVector is a histogram of words which is reduced in dimensionality with
// a random transform
type BigVector struct {
	// Vector is a dimensionality reduced histogram of words, so this vector
	// is a document vector
	Vector []int64
	// Words is a hash table of words mapped to vectors
	// the vectors are dimensionally reduced histograms of words found
	// near a particular word, so the vectors are word vectors
	Words map[string][]int64
	// Name the name of this document vector
	Name string
}

// NewBigVector creates a new big vector
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

// ProcessFile processes a file and computes the document vector and word
// vectors
func (b *BigVector) ProcessFile(name string) {
	file, err := os.Open(name)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	vector, cache, reader, word, buffer, size :=
		b.Vector, make(map[uint64][]int8), bufio.NewReader(file), "", NewCircularBuffer(), len(b.Vector)

	// lookup a cached transform
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
			// compute the order 1 markov model document vector
			transform := lookup(buffer.GetPrevious() + word)
			for i, t := range transform {
				vector[i] += int64(t)
			}

			// find the word vector for the current word
			center := buffer.Item(bufferSize / 2)
			wordVector := b.Words[center]
			if wordVector == nil {
				wordVector = make([]int64, size)
				b.Words[center] = wordVector
			}

			// compute the word vector
			/*for i := 0; i < bufferSize; i++ {
				current := buffer.Lookup(i)
				if current == center {
					continue
				}
				transform := lookup(current)
				for i, t := range transform {
					wordVector[i] += int64(t)
				}
			}*/

			// compute the order 1 markov model word vector
			last := buffer.Item(0)
			for i := 1; i < bufferSize; i++ {
				current := buffer.Item(i)
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

// Distance computes the distance between two document vectors
func (b *BigVector) Distance(a *BigVector) float64 {
	/*var d int64
		for i, j := range b.Vector {
			diff := j - a.Vector[i]
			d += diff * diff
		}
	  return float64(d)*/
	return Similarity(a.Vector, b.Vector)
}

// Similarity computes the distance between two vectors
func Similarity(a, b []int64) float64 {
	dot, xx, yy := 0.0, 0.0, 0.0
	for i, j := range b {
		x, y := float64(a[i]), float64(j)
		dot += x * y
		xx += x * x
		yy += y * y
	}
	return dot / math.Sqrt(xx*yy)
}

// Distance represents the distance between a query document and another
// docuemnt
type Distance struct {
	D    float64
	Name string
}

// Distances is a sortable slice of distances
type Distances []Distance

// Len is the length of the Distances slice
func (d Distances) Len() int {
	return len(d)
}

// Swap swaps two items in the slice
func (d Distances) Swap(i, j int) {
	d[i], d[j] = d[j], d[i]
}

// Less determines if one distance is less than another distance
func (d Distances) Less(i, j int) bool {
	return d[i].D > d[j].D
}

func main() {
	// process the files in data in a parallelized fasion
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

	// sum the word vectors across all documents
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

	// sort the documents by how well they match the query document
	fmt.Println("\ndocument match:")
	distances := make(Distances, len(files))
	for i := range distances {
		distances[i].D = query.Distance(vectors[i])
		distances[i].Name = vectors[i].Name
	}
	sort.Sort(distances)
	for d := range distances {
		fmt.Printf("%v, %v\n", authors[distances[d].Name], distances[d].Name)
	}

	// find words that match the query word
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
