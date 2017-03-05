--[[
            Torch translation of the keras example found here (written by Eder Santana).
            https://gist.github.com/EderSantana/c7222daa328f0e885093#file-qlearn-py-L164

            Example of Re-inforcement learning using the Q function described in this paper from deepmind.
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

            The agent plays a game of catch. Fruits drop from the sky and the agent can choose the actions
            left/stay/right to catch the fruit before it reaches the ground.
]] --

require 'nn'
require 'CatchEnvironment'
require 'optim'

local cmd = torch.CmdLine()
cmd:text('Training options')
cmd:option('-epsilon', 1, 'The probability of choosing a random action (in training). This decays as iterations increase. (0 to 1)')
cmd:option('-epsilonMinimumValue', 0.001, 'The minimum value we want epsilon to reach in training. (0 to 1)')
cmd:option('-nbActions', 3, 'The number of actions. Since we only have left/stay/right that means 3 actions.')
cmd:option('-epoch', 1000, 'The number of games we want the system to run for.')
cmd:option('-hiddenSize', 100, 'Number of neurons in the hidden layers.')
cmd:option('-maxMemory', 500, 'How large should the memory be (where it stores its past experiences).')
cmd:option('-batchSize', 50, 'The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.')
cmd:option('-gridSize', 10, 'The size of the grid that the agent is going to play the game on.')
cmd:option('-discount', 0.9, 'the discount is used to force the network to choose states that lead to the reward quicker (0 to 1)')
cmd:option('-savePath', 'TorchQLearningModel.t7', 'Save path for model')
cmd:option('-learningRate', 0.1)
cmd:option('-learningRateDecay', 1e-9)
cmd:option('-weightDecay', 0)
cmd:option('-momentum', 0.9)

local opt = cmd:parse(arg)

local epsilon = opt.epsilon
local epsilonMinimumValue = opt.epsilonMinimumValue
local nbActions = opt.nbActions
local epoch = opt.epoch
local hiddenSize = opt.hiddenSize
local maxMemory = opt.maxMemory
local batchSize = opt.batchSize
local gridSize = opt.gridSize
local nbStates = gridSize * gridSize
local discount = opt.discount

math.randomseed(os.time())

--[[ Helper function: Chooses a random value between the two boundaries.]] --
local function randf(s, e)
    return (math.random(0, (e - s) * 9999) / 10000) + s;
end

--[[ The memory: Handles the internal memory that we add experiences that occur based on agent's actions,
--   and creates batches of experiences based on the mini-batch size for training.]] --
local function Memory(maxMemory, discount)
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

            local target = model:forward(memoryInput.inputState):clone()

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
        local gradOutput = criterion:backward(predictions, targets)
        model:backward(inputs, gradOutput)
        return loss, gradParameters
    end

    local _, fs = optim.sgd(feval, x, sgdParams)
    loss = loss + fs[1]
    return loss
end

-- Create the base model.
local model = nn.Sequential()
model:add(nn.Linear(nbStates, hiddenSize))
model:add(nn.ReLU())
model:add(nn.Linear(hiddenSize, hiddenSize))
model:add(nn.ReLU())
model:add(nn.Linear(hiddenSize, nbActions))

-- Params for Stochastic Gradient Descent (our optimizer).
local sgdParams = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    dampening = 0,
    nesterov = true
}

-- Mean Squared Error for our loss function.
local criterion = nn.MSECriterion()

local env = CatchEnvironment(gridSize)
local memory = Memory(maxMemory, discount)

local winCount = 0
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
        if (reward == 1) then winCount = winCount + 1 end
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
    print(string.format("Epoch %d : err = %f : Win count %d ", i, err, winCount))
end
torch.save(opt.savePath, model)
print("Model saved to " .. opt.savePath)
