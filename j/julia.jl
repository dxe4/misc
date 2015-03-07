using Images
using Color
# modification of https://github.com/one-more-minute/Juno-LT/blob/master/tutorial.jl
const ϕ = golden
const iterations = 200

cm = colormap("oranges", iterations)
function julia(z)
  c = (φ-2)+(φ-1)im
  max = iterations
  for n = 1:max
    if abs(z) ≥ 2
        zn = (z.re * z.re + z.im * z.im) ^ 2
        nu = log(log(n)) / log(2)
        a = int((n + 1 - nu) / max * 255)
        if a > iterations
          return n-1
        else
          return a
        end
    end
    z = z ^ 2 + c
  end
  return max
end


julia(x, y) = julia(x + y*im)

julia_grid(n) =
  broadcast(julia,
            linspace(-0.4, 0.6, n)',
            linspace(-0.4, 0.6, n))


a = julia_grid(4000)

new_array =  Array(RGB{Float64}, 4000, 4000)

for i = 1:size(a, 1)
  for j = 1:size(a, 2)
    index = int(a[i, j])
    new_array[i,j] = cm[index]
  end
end
Images.imwrite(new_array, "foo.png")
