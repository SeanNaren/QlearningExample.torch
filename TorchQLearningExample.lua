--[[
            Example of Re-inforcement learning using the Q function described in this paper from deepmind.
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

            The example itself is very simple. The agent must navigate an array where the agent can move left or right.
            The position of the agent is given by the 1 and the ends of the array cycle (so one move right from 9 goes to 0).
            [1,0,0,0,0,0,0,0,0]
            There is one reward found in the middle (since there is 9 spaces, we round up, so a reward at 5) that the
            agent must navigate to. It is rewarded for reaching that state.
]] --

require 'nn'
require 'optim'

math.randomseed(os.time())

--[[ Helper function: Chooses a random value between the two boundaries.]] --
local function randf(s, e)
    return (math.random(0, (e - s) * 9999) / 10000) + s;
end

--[[ The environment: Handles interactions and contains the state of the environment]] --
local function createEnv(nbStates)
    local env = {}
    local state
    local currentPositionOfAgent = 1
    -- Returns the state of the environment.
    function env.observe()
        return state
    end

    -- Resets the environment back to the initial set up.
    function env.reset()
        state = torch.Tensor(nbStates):zero()
        currentPositionOfAgent = 1
        state[currentPositionOfAgent] = 1 -- The initial position of the agent we set to 1.
    end

    -- Returns true if user is at the reward state.
    function env.finishingMove()
        return currentPositionOfAgent == math.floor(nbStates / 2)
    end

    -- Returns the award that the agent has gained for being in the current environment state.
    function env.getReward()
        local reward
        -- There is one state (roughly half way depending on the number of states) that has a reward 1.
        -- Rest of the states are rewards of 0.
        if (env.finishingMove()) then
            reward = 1
        else
            reward = 0
        end
        return reward
    end

    function env.isGameOver()
        -- If it was a finishing move then we end the game.
        -- Since in this example there is only one reward, we also say that this is the finishing stage.
        local gameOver = env.finishingMove()
        return gameOver
    end

    -- Action can be 1 (move left) or 2 (move right)
    function env.act(action)
        if (action == 1) then
            -- If the current state of the agent is at the start, we set the agent's position to the end (cyclic).
            if (currentPositionOfAgent == 1) then
                currentPositionOfAgent = nbStates
            else
                currentPositionOfAgent = currentPositionOfAgent - 1
            end
        else
            -- If the current state of the agent is at the end, we set the agent's position to the start (cyclic).
            if (currentPositionOfAgent == nbStates) then
                currentPositionOfAgent = 1
            else
                currentPositionOfAgent = currentPositionOfAgent + 1
            end
        end
        -- Update the state of the environment now that the agent has moved.
        state = torch.Tensor(nbStates):zero()
        state[currentPositionOfAgent] = 1
        local reward = env.getReward()
        local gameOver = env.isGameOver()
        return state, reward, gameOver
    end

    --We initialise the environment.
    env.reset()
    return env
end

--[[ The memory: Handles the internal memory that we add experiences that occur based on agent's actions,
--   and creates batches of experiences based on the mini-batch size for training.]] --
local function createMemory(maxMemory, discount)
    local memory = {}

    -- Appends the experience to the memory.
    function memory.remember(memoryInput)
        table.insert(memory, memoryInput)
        if (#memory > maxMemory) then
            -- Remove the earliest memory to allocate new experience to memory.
            table.remove(memory, 1)
        end
    end

    function memory.getBatch(model, batchSize, nbActions, nbStates)

        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch).
        local memoryLength = #memory
        local chosenBatchSize = math.min(batchSize, memoryLength)

        local inputs = torch.Tensor(chosenBatchSize, nbStates):zero()
        local targets = torch.Tensor(chosenBatchSize, nbActions):zero()

        --Fill the inputs and targets up.
        for i = 1, chosenBatchSize do
            -- Choose a random memory experience to add to the batch.
            local randomIndex = math.random(1, memoryLength)
            local memoryInput = memory[randomIndex]

            local target = model:forward(memoryInput.inputState)

            --Gives us Q_sa, the max q for the next state.
            local nextStateMaxQ = torch.max(model:forward(memoryInput.nextState), 1)[1]
            if (memoryInput.gameOver) then
                target[memoryInput.action] = memoryInput.reward
            else
                -- reward + discount(gamma) * max_a' Q(s',a')
                -- We are setting the Q-value for the action to  r + γmax a’ Q(s’, a’). The rest stay the same
                -- to give an error of 0 for those outputs.
                target[memoryInput.action] = memoryInput.reward + discount * nextStateMaxQ
            end
            -- Update the inputs and targets.
            inputs[i] = memoryInput.inputState
            targets[i] = target
        end
        return inputs, targets
    end

    return memory
end

--[[ Runs one gradient update using SGD returning the loss.]] --
local function trainNetwork(model, inputs, targets, criterion, sgdParams)
    local loss = 0
    local x, gradParameters = model:getParameters()
    local function feval(x_new)
        gradParameters:zero()
        local predictions = model:forward(inputs)
        local loss = criterion:forward(predictions, targets)
        model:zeroGradParameters()
        local gradOutput = criterion:backward(predictions, targets)
        model:backward(inputs, gradOutput)
        return loss, gradParameters
    end

    local _, fs = optim.sgd(feval, x, sgdParams)
    loss = loss + fs[1]
    return loss
end

local epsilon = 0.3 -- The probability of choosing a random action (in training). This decays as iterations increase. (0 to 1)
local epsilonMinimumValue = 0.1 -- The minimum value we want epsilon to reach in training. (0 to 1)
local nbActions = 2 -- The number of actions. Since we only have left/right that means 2 actions.
local epoch = 500 -- The number of games we want the system to run for.
local maxMemory = 500 -- How large should the memory be (where it stores its past experiences).
local batchSize = 50 -- The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.
local nbStates = 9 -- The number of states in the system in terms of the environment. In our case this is 9.
local discount = 0.9 -- The discount is used to force the network to choose states that lead to the reward quicker (0 to 1)

-- Create the base model.
local model = nn.Sequential()
model:add(nn.Linear(nbStates, 100))
model:add(nn.ReLU())
model:add(nn.Linear(100, 100))
model:add(nn.ReLU())
model:add(nn.Linear(100, nbActions))

-- Params for Stochastic Gradient Descent (our optimizer).
local sgdParams = {
    learningRate = 0.0002,
    learningRateDecay = 1e-9,
    weightDecay = 0,
    momentum = 0.9,
    dampening = 0,
    nesterov = true
}

-- Mean Squared Error for our loss function.
local criterion = nn.MSECriterion()

local env = createEnv(nbStates)
local memory = createMemory(maxMemory, discount)

for i = 1, epoch do
    -- Initialise the environment.
    local err = 0
    env.reset()
    local isGameOver = false

    -- The initial state of the environment.
    local currentState = env.observe()

    while (isGameOver ~= true) do
        local action
        -- Decides if we should choose a random action, or an action from the policy network.
        if (randf(0, 1) <= epsilon) then
            action = math.random(1, nbActions)
        else
            -- Forward the current state through the network.
            local q = model:forward(currentState)
            -- Find the max index (the chosen action).
            local max, index = torch.max(q, 1)
            action = index[1]
        end
        -- Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
        if (epsilon > epsilonMinimumValue) then
            epsilon = epsilon * 0.999
        end
        local nextState, reward, gameOver = env.act(action)
        memory.remember({
            inputState = currentState,
            action = action,
            reward = reward,
            nextState = nextState,
            gameOver = gameOver
        })
        -- Update the current state and if the game is over.
        currentState = nextState
        isGameOver = gameOver

        -- We get a batch of training data to train the model.
        local inputs, targets = memory.getBatch(model, batchSize, nbActions, nbStates)

        -- Train the network which returns the error.
        err = err + trainNetwork(model, inputs, targets, criterion, sgdParams)
    end
    print(string.format("Epoch %d : err = %f ", i, err))
end


env.reset()
local isGameOver = false
local inputState = env.observe()

while (isGameOver ~= true) do
    -- Forward the current state through the network.
    local q = model:forward(inputState)
    -- Find the max index (the chosen action).
    local max, index = torch.max(q, 1)
    local action = index[1]
    print(action)
    local nextState, reward, gameOver = env.act(action)
    inputState = nextState
    isGameOver = gameOver
end

