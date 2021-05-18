using Plots

function grid_with_boundary(k)
    n = 2^k

    # Grid
    f = zeros(n+1,n+1)

    # Boundary Conditions:
    
    # Bottom
    f[1,1:(n÷2 + 1)] .= range(13,5;length= n ÷ 2 + 1)
    f[1, (n÷2 + 1) : (3*n÷4)] .= 5
    f[1, (3*n÷4+1):(n+1)] .= range(5,13;length=n÷4+1)

    # Top
    f[n+1,1:n+1] .= 21

    # Left
    f[1:(3*n÷8+1),1] .= range(13,40;length=3*n÷8+1)
    f[(n÷2+1):n+1,1] .= range(10,21;length=n÷2+1)

    # Right
    f[1:(n÷2+1),n+1] .= range(13,40;length=n÷2+1)
    f[(5*n÷8+1):n+1, n+1] .= range(40,21;length=3*n÷8+1)

    # Heaters
    f[(3*n÷8+1):(n÷2+1),1:(n÷8+1)] .= 40
    f[(n÷2+1):(5*n÷8+1), (7*n÷8+1):n+1] .= 40

    return f
end


function simulation(T, ϵ=1e-3, num_steps=200)
    results = [T]
    for i in 1:num_steps
        T' = jacobi_step(T)
        add_heaters!(T')

        push!(results,copy(T'))

        if maximum(abs.(T' - T)) < ϵ
            return results
        end

        T = copy(T')
    end
    return results
end


function jacobi_step(T::AbstractMatrix)
    m,n = size(T)
    T' = deepcopy(T)

    # iterate over interior 
    for i in 2:m-1, j in 2:n-1
        T'[i,j] = (T[i+1,j] + T[i-1,j] + T[i,j-1] + T[i,j+1]) / 4
    end

    return T'
end

T = grid_with_boundary(5)


myheatmap(T) = heatmap(T,yflip=true, ratio=1)
@show myheatmap(results[i])

function initial_condition(T)
    m,n = size(T)
    T' = copy(T)
    for i in 2:m-1
        for j in 2:n-1
            T′[i, j] = ( j * T[i, n] + (n + 1 - j) * T[i, 1] + 
					(m + 1 - i) * T[1, j] + i * T[m, j]) / (m + n + 2)
        end
    end
    return T'
end

TT = grid_with_boundary(10)
TT' = initial_condition(TT)

add_heaters!(TT')
