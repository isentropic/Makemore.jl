module Makemore
using Flux

function loaddata(filename)
    lines = map(strip, filename |> open |> readlines)
    uniquechars = Set{Char}()
    for line in lines
        for c in line
            push!(uniquechars, c)
        end
    end

    uniquechars_count = len(uniquechars)
end

end
