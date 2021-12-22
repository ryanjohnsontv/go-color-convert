package colorconvert

import (
	"errors"
	"math"
	"strconv"
	"strings"
)

var COLORS = map[string][]int{
	"aliceblue":            {240, 248, 255},
	"antiquewhite":         {250, 235, 215},
	"aqua":                 {0, 255, 255},
	"aquamarine":           {127, 255, 212},
	"azure":                {240, 255, 255},
	"beige":                {245, 245, 220},
	"bisque":               {255, 228, 196},
	"black":                {0, 0, 0},
	"blanchedalmond":       {255, 235, 205},
	"blue":                 {0, 0, 255},
	"blueviolet":           {138, 43, 226},
	"brown":                {165, 42, 42},
	"burlywood":            {222, 184, 135},
	"cadetblue":            {95, 158, 160},
	"chartreuse":           {127, 255, 0},
	"chocolate":            {210, 105, 30},
	"coral":                {255, 127, 80},
	"cornflowerblue":       {100, 149, 237},
	"cornsilk":             {255, 248, 220},
	"crimson":              {220, 20, 60},
	"cyan":                 {0, 255, 255},
	"darkblue":             {0, 0, 139},
	"darkcyan":             {0, 139, 139},
	"darkgoldenrod":        {184, 134, 11},
	"darkgray":             {169, 169, 169},
	"darkgreen":            {0, 100, 0},
	"darkgrey":             {169, 169, 169},
	"darkkhaki":            {189, 183, 107},
	"darkmagenta":          {139, 0, 139},
	"darkolivegreen":       {85, 107, 47},
	"darkorange":           {255, 140, 0},
	"darkorchid":           {153, 50, 204},
	"darkred":              {139, 0, 0},
	"darksalmon":           {233, 150, 122},
	"darkseagreen":         {143, 188, 143},
	"darkslateblue":        {72, 61, 139},
	"darkslategray":        {47, 79, 79},
	"darkslategrey":        {47, 79, 79},
	"darkturquoise":        {0, 206, 209},
	"darkviolet":           {148, 0, 211},
	"deeppink":             {255, 20, 147},
	"deepskyblue":          {0, 191, 255},
	"dimgray":              {105, 105, 105},
	"dimgrey":              {105, 105, 105},
	"dodgerblue":           {30, 144, 255},
	"firebrick":            {178, 34, 34},
	"floralwhite":          {255, 250, 240},
	"forestgreen":          {34, 139, 34},
	"fuchsia":              {255, 0, 255},
	"gainsboro":            {220, 220, 220},
	"ghostwhite":           {248, 248, 255},
	"gold":                 {255, 215, 0},
	"goldenrod":            {218, 165, 32},
	"gray":                 {128, 128, 128},
	"green":                {0, 128, 0},
	"greenyellow":          {173, 255, 47},
	"grey":                 {128, 128, 128},
	"honeydew":             {240, 255, 240},
	"hotpink":              {255, 105, 180},
	"indianred":            {205, 92, 92},
	"indigo":               {75, 0, 130},
	"ivory":                {255, 255, 240},
	"khaki":                {240, 230, 140},
	"lavender":             {230, 230, 250},
	"lavenderblush":        {255, 240, 245},
	"lawngreen":            {124, 252, 0},
	"lemonchiffon":         {255, 250, 205},
	"lightblue":            {173, 216, 230},
	"lightcoral":           {240, 128, 128},
	"lightcyan":            {224, 255, 255},
	"lightgoldenrodyellow": {250, 250, 210},
	"lightgray":            {211, 211, 211},
	"lightgreen":           {144, 238, 144},
	"lightgrey":            {211, 211, 211},
	"lightpink":            {255, 182, 193},
	"lightsalmon":          {255, 160, 122},
	"lightseagreen":        {32, 178, 170},
	"lightskyblue":         {135, 206, 250},
	"lightslategray":       {119, 136, 153},
	"lightslategrey":       {119, 136, 153},
	"lightsteelblue":       {176, 196, 222},
	"lightyellow":          {255, 255, 224},
	"lime":                 {0, 255, 0},
	"limegreen":            {50, 205, 50},
	"linen":                {250, 240, 230},
	"magenta":              {255, 0, 255},
	"maroon":               {128, 0, 0},
	"mediumaquamarine":     {102, 205, 170},
	"mediumblue":           {0, 0, 205},
	"mediumorchid":         {186, 85, 211},
	"mediumpurple":         {147, 112, 219},
	"mediumseagreen":       {60, 179, 113},
	"mediumslateblue":      {123, 104, 238},
	"mediumspringgreen":    {0, 250, 154},
	"mediumturquoise":      {72, 209, 204},
	"mediumvioletred":      {199, 21, 133},
	"midnightblue":         {25, 25, 112},
	"mintcream":            {245, 255, 250},
	"mistyrose":            {255, 228, 225},
	"moccasin":             {255, 228, 181},
	"navajowhite":          {255, 222, 173},
	"navy":                 {0, 0, 128},
	"navyblue":             {0, 0, 128},
	"oldlace":              {253, 245, 230},
	"olive":                {128, 128, 0},
	"olivedrab":            {107, 142, 35},
	"orange":               {255, 165, 0},
	"orangered":            {255, 69, 0},
	"orchid":               {218, 112, 214},
	"palegoldenrod":        {238, 232, 170},
	"palegreen":            {152, 251, 152},
	"paleturquoise":        {175, 238, 238},
	"palevioletred":        {219, 112, 147},
	"papayawhip":           {255, 239, 213},
	"peachpuff":            {255, 218, 185},
	"peru":                 {205, 133, 63},
	"pink":                 {255, 192, 203},
	"plum":                 {221, 160, 221},
	"powderblue":           {176, 224, 230},
	"purple":               {128, 0, 128},
	"red":                  {255, 0, 0},
	"rosybrown":            {188, 143, 143},
	"royalblue":            {65, 105, 225},
	"saddlebrown":          {139, 69, 19},
	"salmon":               {250, 128, 114},
	"sandybrown":           {244, 164, 96},
	"seagreen":             {46, 139, 87},
	"seashell":             {255, 245, 238},
	"sienna":               {160, 82, 45},
	"silver":               {192, 192, 192},
	"skyblue":              {135, 206, 235},
	"slateblue":            {106, 90, 205},
	"slategray":            {112, 128, 144},
	"slategrey":            {112, 128, 144},
	"snow":                 {255, 250, 250},
	"springgreen":          {0, 255, 127},
	"steelblue":            {70, 130, 180},
	"tan":                  {210, 180, 140},
	"teal":                 {0, 128, 128},
	"thistle":              {216, 191, 216},
	"tomato":               {255, 99, 71},
	"turquoise":            {64, 224, 208},
	"violet":               {238, 130, 238},
	"wheat":                {245, 222, 179},
	"white":                {255, 255, 255},
	"whitesmoke":           {245, 245, 245},
	"yellow":               {255, 255, 0},
	"yellowgreen":          {154, 205, 50},
	"homeassistant":        {3, 169, 244},
}

func MinMax(values []float64) (float64, float64) {
	var max float64 = values[0]
	var min float64 = values[0]
	for _, v := range values {
		if max < v {
			max = v
		}
		if min > v {
			min = v
		}
	}
	return min, max
}
func Min(values []float64) float64 {
	var min float64 = values[0]
	for _, v := range values {
		if min > v {
			min = v
		}
	}
	return min
}
func Max(values []float64) float64 {
	var max float64 = values[0]
	for _, v := range values {
		if max < v {
			max = v
		}
	}
	return max
}
func Round(val float64) int {
	if val < 0 {
		return int(val - 0.5)
	}
	return int(val + 0.5)
}
func Ceil(val float64) int {
	if val > 0 {
		return int(val + 1.0)
	}
	return int(val)
}

func Floor(val float64) int {
	if val < 0 {
		return int(val - 1.0)
	}
	return int(val)
}
func Round3(values []float64) []int {
	newvals := make([]int, len(values))
	for i, v := range values {
		newvals[i] = Round(v)
	}
	return newvals
}

func rgb_to_hsl(rgb []int) []int {
	var r, g, b, h, s float64
	r = float64(rgb[0]) / 255
	g = float64(rgb[1]) / 255
	b = float64(rgb[2]) / 255
	min, max := MinMax([]float64{r, g, b})
	delta := max - min

	if max == min {
		h = 0
	} else if r == max {
		h = (g - b) / delta
	} else if g == max {
		h = 2 + (b-r)/delta
	} else if b == max {
		h = 4 + (r-g)/delta
	}

	h, _ = MinMax([]float64{h * 60, 360})

	if h < 0 {
		h += 360
	}

	l := (min + max) / 2

	if max == min {
		s = 0
	} else if l <= 0.5 {
		s = delta / (max + min)
	} else {
		s = delta / (2 - max - min)
	}
	return Round3([]float64{h, (s * 100), (l * 100)})
	//return []int{Round(h), Round(s * 100), Round(l * 100)}
}

func rgb_to_hsv(rgb []int) []int {
	var r, g, b, h, s, rdif, gdif, bdif float64
	r = float64(rgb[0]) / 255
	g = float64(rgb[1]) / 255
	b = float64(rgb[2]) / 255
	min, v := MinMax([]float64{r, g, b})
	delta := v - min

	if delta == 0 {
		h = 0
		s = 0
	} else {
		s = delta / v
		rdif = (v-r)/6/delta + 1/2
		gdif = (v-g)/6/delta + 1/2
		bdif = (v-b)/6/delta + 1/2

		if r == v {
			h = bdif - gdif
		} else if g == v {
			h = (1 / 3) + rdif - bdif
		} else if b == v {
			h = (2 / 3) + gdif - rdif
		}

		if h < 0 {
			h += 1
		} else if h > 1 {
			h -= 1
		}
	}
	return Round3([]float64{(h * 360), (s * 100), (v * 100)})
	//return []int{Round(h * 360), Round(s * 100), Round(v * 100)}
}

func rgb_to_hwb(rgb []int) []int {
	r := float64(rgb[0])
	g := float64(rgb[1])
	b := float64(rgb[2])
	h := rgb_to_hsl(rgb)[0]
	w := 1 / 255 * Min([]float64{r, Min([]float64{g, b})})

	b = 1 - 1/255*Max([]float64{r, Max([]float64{g, b})})

	return Round3([]float64{float64(h), (w * 100), (b * 100)})
	//return []int{h, Round(w * 100), Round(b * 100)}
}

func rgb_to_cmyk(rgb []int) []int {
	r := float64(rgb[0]) / 255
	g := float64(rgb[1]) / 255
	b := float64(rgb[2]) / 255

	k := Min([]float64{1 - r, 1 - g, 1 - b})
	c := Round((1 - r - k) / (1 - k))
	m := Round((1 - g - k) / (1 - k))
	y := Round((1 - b - k) / (1 - k))

	return []int{c * 100, m * 100, y * 100, Round(k * 100)}
}

func comparativeDistance(x []int, y []int) int {
	/*
		See https://en.m.wikipedia.org/wiki/Euclidean_distance#Squared_Euclidean_distance
	*/
	return Round(math.Pow(float64(x[0]-y[0]), 2) + math.Pow(float64(x[1]-y[1]), 2) + math.Pow(float64(x[2]-y[2]), 2))
}

func rgb_to_keyword(rgb []int) string {
	for keyword, value := range COLORS {
		if rgb[0] == value[0] && rgb[1] == value[1] && rgb[2] == value[2] {
			return keyword
		}
	}

	var currentClosestDistance int
	var currentClosestKeyword string

	for keyword, value := range COLORS {
		// Compute comparative distance
		distance := comparativeDistance(rgb, value)

		// Check if its less, if so set as closest
		if distance < currentClosestDistance {
			currentClosestDistance = distance
			currentClosestKeyword = keyword
		}
	}

	return currentClosestKeyword
}

func color_name_to_rgb(color_name string) ([]int, error) {
	keyword := strings.ToLower(strings.ReplaceAll(color_name, " ", ""))
	//Convert color name to RGB hex value.
	if rgb, ok := COLORS[keyword]; ok {
		return rgb, nil
	}
	return nil, errors.New("Unknown Color Name")
}

func rgb_to_xyz(rgb []int) []int {
	r := float64(rgb[0]) / 255
	g := float64(rgb[1]) / 255
	b := float64(rgb[2]) / 255

	// Assume sRGB
	if r > 0.04045 {
		r = math.Pow((r + 0.055), 2.4)
	} else {
		r = (r / 12.92)
	}
	if b > 0.04045 {
		b = math.Pow((b + 0.055), 2.4)
	} else {
		b = (b / 12.92)
	}
	if g > 0.04045 {
		g = math.Pow((g + 0.055), 2.4)
	} else {
		g = (g / 12.92)
	}

	x := (r * 0.4124564) + (g * 0.3575761) + (b * 0.1804375)
	y := (r * 0.2126729) + (g * 0.7151522) + (b * 0.072175)
	z := (r * 0.0193339) + (g * 0.119192) + (b * 0.9503041)

	return Round3([]float64{(x * 100), (y * 100), (z * 100)})
	//return []int{Round(x * 100), Round(y * 100), Round(z * 100)}
}

func rgb_to_lab(rgb []int) []int {
	xyz := rgb_to_xyz(rgb)
	x := float64(xyz[0])
	y := float64(xyz[1])
	z := float64(xyz[2])

	x /= 95.047
	y /= 100
	z /= 108.883

	if x > 0.008856 {
		x = math.Pow(x, (1 / 3))
	} else {
		x = (7.787 * x) + (16 / 116)
	}
	if y > 0.008856 {
		y = math.Pow(y, (1 / 3))
	} else {
		y = (7.787 * y) + (16 / 116)
	}
	if z > 0.008856 {
		z = math.Pow(z, (1 / 3))
	} else {
		z = (7.787 * z) + (16 / 116)
	}

	l := (116 * y) - 16
	a := 500 * (x - y)
	b := 200 * (y - z)

	return Round3([]float64{l, a, b})
	//return []int{Round(l), Round(a), Round(b)}
}

func hsl_to_rgb(hsl []int) []int {
	h := float64(hsl[0] / 360)
	s := float64(hsl[1] / 100)
	l := float64(hsl[2] / 100)
	var t2, t3, val float64

	if s == 0 {
		val = l * 255
		return []int{Round(val), Round(val), Round(val)}
	}

	if l < 0.5 {
		t2 = l * (1 + s)
	} else {
		t2 = l + s - l*s
	}

	t1 := 2*l - t2

	rgb := []int{0, 0, 0}
	for i := 0; i < 3; i++ {
		t3 = h + float64(1/3*-(i-1))
		if t3 < 0 {
			t3++
		}

		if t3 > 1 {
			t3--
		}

		if 6*t3 < 1 {
			val = t1 + (t2-t1)*6*t3
		} else if 2*t3 < 1 {
			val = t2
		} else if 3*t3 < 2 {
			val = t1 + (t2-t1)*(2/3-t3)*6
		} else {
			val = t1
		}

		rgb[i] = Round(val * 255)
	}

	return rgb
}

func hsl_to_hsv(hsl []int) []int {
	h := float64(hsl[0])
	s := float64(hsl[1] / 100)
	l := float64(hsl[2] / 100)
	smin := s
	lmin := Max([]float64{l, 0.01})

	l *= 2
	if l <= 1 {
		s *= l
	} else {
		s *= 2 - l
	}
	if lmin <= 1 {
		smin *= lmin
	} else {
		smin *= 2 - lmin
	}
	v := (l + s) / 2

	var sv float64
	if l == 0 {
		sv = (2 * smin) / (lmin + smin)
	} else {
		sv = (2 * s) / (l + s)
	}

	return Round3([]float64{h, (sv * 100), (v * 100)})
	//return []int{Round(h), Round(sv * 100), Round(v * 100)}
}

func hsv_to_rgb(hsv []int) []int {
	h := float64(hsv[0] / 60)
	s := float64(hsv[1] / 100)
	v := float64(hsv[2] / 100)
	hi := Floor(h) % 6

	f := h - float64(Floor(h))
	p := Round(255 * v * (1 - s))
	q := Round(255 * v * (1 - (s * f)))
	t := Round(255 * v * (1 - (s * (1 - f))))
	v *= 255

	switch hi {
	case 0:
		return []int{Round(v), t, p}
	case 1:
		return []int{q, Round(v), p}
	case 2:
		return []int{p, Round(v), t}
	case 3:
		return []int{p, q, Round(v)}
	case 4:
		return []int{t, p, Round(v)}
	case 5:
		return []int{Round(v), p, q}
	default:
		return []int{255, 255, 255}
	}
}

func hsv_to_hsl(hsv []int) []int {
	h := float64(hsv[0])
	s := float64(hsv[1] / 100)
	v := float64(hsv[2] / 100)
	vmin := Max([]float64{v, 0.01})

	l := (2 - s) * v
	lmin := (2 - s) * vmin
	sl := s * vmin
	if lmin <= 1 {
		sl /= lmin
	} else {
		sl /= 2 - lmin
	}
	l /= 2

	return Round3([]float64{h, (sl * 100), (l * 100)})
	//return []int{Round(h), Round(sl * 100), Round(l * 100)}
}

// http://dev.w3.org/csswg/css-color/#hwb-to-rgb
func hwb_to_rgb(hwb []int) []int {
	h := float64(hwb[0] / 360)
	wh := float64(hwb[1] / 100)
	bl := float64(hwb[2] / 100)
	ratio := wh + bl

	// Wh + bl cant be > 1
	if ratio > 1 {
		wh /= ratio
		bl /= ratio
	}

	i := Floor(6 * h)
	v := 1 - bl
	f := 6*h - float64(i)

	if (i & 0x01) != 0 {
		f = 1 - f
	}

	n := wh + f*(v-wh) // Linear interpolation

	var r, g, b float64

	// eslint-disable max-statements-per-line,no-multi-spaces
	switch i {
	default:
	case 6:
	case 0:
		r = v
		g = n
		b = wh
		break
	case 1:
		r = n
		g = v
		b = wh
		break
	case 2:
		r = wh
		g = v
		b = n
		break
	case 3:
		r = wh
		g = n
		b = v
		break
	case 4:
		r = n
		g = wh
		b = v
		break
	case 5:
		r = v
		g = wh
		b = n
		break
	}
	// eslint-enable max-statements-per-line,no-multi-spaces
	return Round3([]float64{(r * 255), (g * 255), (b * 255)})
	//return []int{Round(r * 255), Round(g * 255), Round(b * 255)}
}

func cmyk_to_rgb(cmyk []int) []int {
	c := float64(cmyk[0] / 100)
	m := float64(cmyk[1] / 100)
	y := float64(cmyk[2] / 100)
	k := float64(cmyk[3] / 100)

	r := 1 - Min([]float64{1, c*(1-k) + k})
	g := 1 - Min([]float64{1, m*(1-k) + k})
	b := 1 - Min([]float64{1, y*(1-k) + k})

	return Round3([]float64{(r * 255), (g * 255), (b * 255)})
	//return []int{Round(r * 255), Round(g * 255), Round(b * 255)}
}

func xyz_to_rgb(xyz []int) []int {
	x := float64(xyz[0] / 100)
	y := float64(xyz[1] / 100)
	z := float64(xyz[2] / 100)
	var r, g, b float64

	r = (x * 3.2404542) + (y * -1.5371385) + (z * -0.4985314)
	g = (x * -0.969266) + (y * 1.8760108) + (z * 0.041556)
	b = (x * 0.0556434) + (y * -0.2040259) + (z * 1.0572252)

	// Assume sRGB
	if r > 0.0031308 {
		r = (1.055 * (math.Pow(r, (1.0 / 2.4)))) - 0.055
	} else {
		r = r * 12.92
	}
	if g > 0.0031308 {
		g = (1.055 * (math.Pow(g, (1.0 / 2.4)))) - 0.055
	} else {
		g = g * 12.92
	}

	if b > 0.0031308 {
		b = (1.055 * (math.Pow(b, (1.0 / 2.4)))) - 0.055
	} else {
		b = b * 12.92
	}

	r = Min([]float64{Max([]float64{0, r}), 1})
	g = Min([]float64{Max([]float64{0, g}), 1})
	b = Min([]float64{Max([]float64{0, b}), 1})

	return Round3([]float64{(r * 255), (g * 255), (b * 255)})
}

func xyz_to_lab(xyz []int) []int {
	x := float64(xyz[0])
	y := float64(xyz[1])
	z := float64(xyz[2])

	x /= 95.047
	y /= 100
	z /= 108.883

	if x > 0.008856 {
		x = math.Pow(x, (1 / 3))
	} else {
		x = (7.787 * x) + (16 / 116)
	}
	if y > 0.008856 {
		y = math.Pow(y, (1 / 3))
	} else {
		y = (7.787 * y) + (16 / 116)
	}
	if z > 0.008856 {
		z = math.Pow(z, (1 / 3))
	} else {
		z = (7.787 * z) + (16 / 116)
	}

	l := (116 * y) - 16
	a := 500 * (x - y)
	b := 200 * (y - z)

	return Round3([]float64{l, a, b})
	//return []int{Round(l), Round(a), Round(b)}
}

func lab_to_xyz(lab []int) []int {
	l := float64(lab[0])
	a := float64(lab[1])
	b := float64(lab[2])
	var x, y, z float64

	y = (l + 16) / 116
	x = a/500 + y
	z = y - b/200

	y2 := math.Pow(y, 3)
	x2 := math.Pow(x, 3)
	z2 := math.Pow(z, 3)
	if y2 > 0.008856 {
		y = y2
	} else {
		y = (y - 16/116) / 7.787
	}
	if x2 > 0.008856 {
		x = x2
	} else {
		x = (x - 16/116) / 7.787
	}
	if z2 > 0.008856 {
		z = z2
	} else {
		z = (z - 16/116) / 7.787
	}

	x *= 95.047
	y *= 100
	z *= 108.883

	return Round3([]float64{x, y, z})
	//return []int{Round(x), Round(y), Round(z)}
}

func lab_to_lch(lab []int) []int {
	l := float64(lab[0])
	a := float64(lab[1])
	b := float64(lab[2])

	hr := math.Atan2(b, a)
	h := hr * 360 / 2 / math.Pi

	if h < 0 {
		h += 360
	}

	c := math.Sqrt(a*a + b*b)

	return Round3([]float64{l, c, h})
	//return []int{Round(l), Round(c), Round(h)}
}

func lch_to_lab(lch []int) []int {
	l := float64(lch[0])
	c := float64(lch[1])
	h := float64(lch[2])

	hr := h / 360 * 2 * math.Pi
	a := c * math.Cos(hr)
	b := c * math.Sin(hr)

	return Round3([]float64{l, a, b})
	//return []int{Round(l), Round(a), Round(b)}
}

/*
func rgb.ansi16(args, saturation = null) {
	[r, g, b] = args;
	let value = saturation === null ? func rgb.hsv(args)[2] : saturation; // Hsv -> ansi16 optimization

	value = Math.Round(value / 50);

	if (value === 0) {
		return 30;
	}

	let ansi = 30
		+ ((Math.Round(b / 255) << 2)
		| (Math.Round(g / 255) << 1)
		| Math.Round(r / 255));

	if (value === 2) {
		ansi += 60;
	}

	return ansi;
};

func hsv.ansi16(args) {
	// Optimization here; we already know the value and don't need to get
	// it converted for us.
	return func rgb.ansi16(func hsv.rgb(args), args[2]);
};

func rgb.ansi256(args) {
	r = args[0];
	g = args[1];
	b = args[2];

	// We use the extended greyscale palette here, with the exception of
	// black and white. normal palette only has 4 greyscale shades.
	if (r >> 4 === g >> 4 && g >> 4 === b >> 4) {
		if (r < 8) {
			return 16;
		}

		if (r > 248) {
			return 231;
		}

		return Math.Round(((r - 8) / 247) * 24) + 232;
	}

	ansi = 16
		+ (36 * Math.Round(r / 255 * 5))
		+ (6 * Math.Round(g / 255 * 5))
		+ Math.Round(b / 255 * 5);

	return ansi;
};

func ansi16.rgb(args) {
	args = args[0];

	let color = args % 10;

	// Handle greyscale
	if (color === 0 || color === 7) {
		if (args > 50) {
			color += 3.5;
		}

		color = color / 10.5 * 255;

		return [color, color, color];
	}

	mult = (~~(args > 50) + 1) * 0.5;
	r = ((color & 1) * mult) * 255;
	g = (((color >> 1) & 1) * mult) * 255;
	b = (((color >> 2) & 1) * mult) * 255;

	return [r, g, b];
};

func ansi256.rgb(args) {
	args = args[0];

	// Handle greyscale
	if (args >= 232) {
		c = (args - 232) * 10 + 8;
		return [c, c, c];
	}

	args -= 16;

	let rem;
	r = Math.floor(args / 36) / 5 * 255;
	g = Math.floor((rem = args % 36) / 6) / 5 * 255;
	b = (rem % 6) / 5 * 255;

	return [r, g, b];
};

func rgb.hex(args) {
	integer = ((Math.Round(args[0]) & 0xFF) << 16)
		+ ((Math.Round(args[1]) & 0xFF) << 8)
		+ (Math.Round(args[2]) & 0xFF);

	string = integer.toString(16).toUpperCase();
	return '000000'.substring(string.length) + string;
};

func hex_to_rgb(args int) []int {
	match = args.toString(16).match(/[a-f0-9]{6}|[a-f0-9]{3}/i);
	if (!match) {
		return [0, 0, 0];
	}

	let colorString = match[0];

	if (match[0].length === 3) {
		colorString = colorString.split('').map(char => {
			return char + char;
		}).join('');
	}

	integer = parseRound(colorString, 16);
	r = (integer >> 16) & 0xFF;
	g = (integer >> 8) & 0xFF;
	b = integer & 0xFF;

	return [r, g, b];
};
*/
func rgb_to_hcg(rgb []int) []int {
	r := float64(rgb[0] / 255)
	g := float64(rgb[1] / 255)
	b := float64(rgb[2] / 255)
	max := Max([]float64{Max([]float64{r, g}), b})
	min := Min([]float64{Min([]float64{r, g}), b})
	chroma := max - min
	var grayscale float64
	var hue int

	if chroma < 1 {
		grayscale = min / (1 - chroma)
	} else {
		grayscale = 0
	}

	if chroma <= 0 {
		hue = 0
	} else if max == r {
		hue = Round(((g - b) / chroma)) % 6
	} else if max == g {
		hue = Round(2 + (b-r)/chroma)
	} else {
		hue = Round(4 + (r-g)/chroma)
	}

	hue /= 6
	hue %= 1

	return []int{hue * 360, Round(chroma * 100), Round(grayscale * 100)}
}

func hsl_to_hcg(hsl []int) []int {
	s := float64(hsl[1] / 100)
	l := float64(hsl[2] / 100)

	var f, c float64
	if l < 0.5 {
		c = (2.0 * s * l)
	} else {
		c = (2.0 * s * (1.0 - l))
	}

	if c < 1.0 {
		f = (l - 0.5*c) / (1.0 - c)
	}

	return []int{hsl[0], Round(c * 100), Round(f * 100)}
}

func hsv_to_hcg(hsv []int) []int {
	s := float64(hsv[1] / 100)
	v := float64(hsv[2] / 100)

	c := s * v
	var f float64

	if c < 1.0 {
		f = (v - c) / (1 - c)
	}

	return []int{hsv[0], Round(c * 100), Round(f * 100)}
}

func hcg_to_rgb(hcg []int) []int {
	h := float64(hcg[0] / 360)
	c := float64(hcg[1] / 100)
	g := float64(hcg[2] / 100)

	if c == 0.0 {
		return Round3([]float64{(g * 255), (g * 255), (g * 255)})
		//return []int{Round(g * 255), Round(g * 255), Round(g * 255)}
	}

	pure := []int{0, 0, 0}
	hi := (Round(h) % 1) * 6
	v := hi % 1
	w := 1 - v
	var mg float64

	// eslint-disable max-statements-per-line
	switch Floor(float64(hi)) {
	case 0:
		pure[0] = 1
		pure[1] = v
		pure[2] = 0
		break
	case 1:
		pure[0] = w
		pure[1] = 1
		pure[2] = 0
		break
	case 2:
		pure[0] = 0
		pure[1] = 1
		pure[2] = v
		break
	case 3:
		pure[0] = 0
		pure[1] = w
		pure[2] = 1
		break
	case 4:
		pure[0] = v
		pure[1] = 0
		pure[2] = 1
		break
	default:
		pure[0] = 1
		pure[1] = 0
		pure[2] = w
	}
	// eslint-enable max-statements-per-line

	mg = (1.0 - c) * g

	return []int{
		Round((c*float64(pure[0]) + mg) * 255),
		Round((c*float64(pure[1]) + mg) * 255),
		Round((c*float64(pure[2]) + mg) * 255),
	}
}

func hcg_to_hsv(hcg []int) []int {
	c := float64(hcg[1] / 100)
	g := float64(hcg[2] / 100)

	v := c + g*(1.0-c)
	var f float64

	if v > 0.0 {
		f = c / v
	}

	return []int{hcg[0], Round(f * 100), Round(v * 100)}
}

func hcg_to_hsl(hcg []int) []int {
	c := float64(hcg[1] / 100)
	g := float64(hcg[2] / 100)

	l := g*(1.0-c) + 0.5*c
	var s float64

	if l > 0.0 && l < 0.5 {
		s = c / (2 * l)
	} else if l >= 0.5 && l < 1.0 {
		s = c / (2 * (1 - l))
	}

	return []int{hcg[0], Round(s * 100), Round(l * 100)}
}

func hcg_to_hwb(hcg []int) []int {
	c := float64(hcg[1] / 100)
	g := float64(hcg[2] / 100)
	v := c + g*(1.0-c)
	return []int{hcg[0], Round((v - c) * 100), Round((1 - v) * 100)}
}

func hwb_to_hcg(hwb []int) []int {
	w := float64(hwb[1] / 100)
	b := float64(hwb[2] / 100)
	v := 1 - b
	c := v - w
	var g float64

	if c < 1 {
		g = (v - c) / (1 - c)
	} else {
		g = 0
	}

	return []int{hwb[0], Round(c * 100), Round(g * 100)}
}

func apple_to_rgb(apple []int) []int {
	return []int{(apple[0] / 65535) * 255, (apple[1] / 65535) * 255, (apple[2] / 65535) * 255}
}

func rgb_to_apple(rgb []int) []int {
	return []int{(rgb[0] / 255) * 65535, (rgb[1] / 255) * 65535, (rgb[2] / 255) * 65535}
}

func gray_to_rgb(args []int) []int {
	return []int{args[0] / 100 * 255, args[0] / 100 * 255, args[0] / 100 * 255}
}

func gray_to_hsl(args []int) []int {
	return []int{0, 0, args[0]}
}

func gray_to_hsv(gray []int) []int {
	return gray_to_hsl(gray)
}

func gray_to_hwb(gray []int) []int {
	return []int{0, 100, gray[0]}
}

func gray_to_cmyk(gray []int) []int {
	return []int{0, 0, 0, gray[0]}
}

func gray_to_lab(gray []int) []int {
	return []int{gray[0], 0, 0}
}

func gray_to_hex(gray []int) string {
	val := Round(float64(gray[0]/100*255)) & 0xFF
	integer := (val << 16) + (val << 8) + val

	hex := strings.ToUpper(strconv.FormatInt(int64(integer), 16))
	return string("000000"[len(hex)]) + hex
}

func rgb_to_gray(rgb []int) []int {
	val := float64((rgb[0] + rgb[1] + rgb[2]) / 3)
	return []int{Round(val / 255 * 100)}
}
