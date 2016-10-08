--[[ The environment: Handles interactions and contains the state of the environment]] --
function CatchEnvironment(gridSize)
    local env = {}
    local state
    -- Returns the state of the environment.
    function env.observe()
        local canvas = env.drawState()
        canvas = canvas:view(-1)
        return canvas
    end

    function env.drawState()
        local canvas = torch.Tensor(gridSize, gridSize):zero()
        canvas[state[1]][state[2]] = 1 -- Draw the fruit.
        -- Draw the basket. The basket takes the adjacent two places to the position of basket.
        canvas[gridSize][state[3] - 1] = 1
        canvas[gridSize][state[3]] = 1
        canvas[gridSize][state[3] + 1] = 1
        return canvas
    end

    -- Resets the environment. Randomly initialise the fruit position (always at the top to begin with) and bucket.
    function env.reset()
        local initialFruitColumn = math.random(1, gridSize)
        local initialBucketPosition = math.random(2, gridSize - 1)
        state = torch.Tensor({ 1, initialFruitColumn, initialBucketPosition })
        return env.getState()
    end

    function env.getState()
        local stateInfo = state
        local fruit_row = stateInfo[1]
        local fruit_col = stateInfo[2]
        local basket = stateInfo[3]
        return fruit_row, fruit_col, basket
    end

    -- Returns the award that the agent has gained for being in the current environment state.
    function env.getReward()
        local fruitRow, fruitColumn, basket = env.getState()
        if (fruitRow == gridSize - 1) then -- If the fruit has reached the bottom.
        if (math.abs(fruitColumn - basket) <= 1) then -- Check if the basket caught the fruit.
        return 1
        else
            return -1
        end
        else
            return 0
        end
    end

    function env.isGameOver()
        if (state[1] == gridSize - 1) then return true else return false end
    end

    function env.updateState(action)
        if (action == 1) then
            action = -1
        elseif (action == 2) then
            action = 0
        else
            action = 1
        end
        local fruitRow, fruitColumn, basket = env.getState()
        local newBasket = math.min(math.max(2, basket + action), gridSize - 1) -- The min/max prevents the basket from moving out of the grid.
        fruitRow = fruitRow + 1 -- The fruit is falling by 1 every action.
        state = torch.Tensor({ fruitRow, fruitColumn, newBasket })
    end

    -- Action can be 1 (move left) or 2 (move right)
    function env.act(action)
        env.updateState(action)
        local reward = env.getReward()
        local gameOver = env.isGameOver()
        return env.observe(), reward, gameOver, env.getState() -- For purpose of the visual, I also return the state.
    end

    return env
end