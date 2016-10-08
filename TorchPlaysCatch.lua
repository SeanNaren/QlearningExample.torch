-- To run this code you must use qtlua, i.e isntead of th use qlua TorchPlaysCatch.lua
require 'nn'
require 'CatchEnvironment'
require 'image'

local cmd = torch.CmdLine()
cmd:text('Training options')
cmd:option('-gridSize', 10, 'The size of the grid that the agent is going to play the game on.')
cmd:option('-modelPath', 'TorchQLearningModel.t7', 'Load path of model')
cmd:option('-maxGames', 100, 'Load path of model')
cmd:option('-renderSize', 512, 'Height and Width to render at')
cmd:option('-backGroundColours', { red = 119, green = 158, blue = 203 }, 'Colour of the background')

local opt = cmd:parse(arg)

local model = torch.load("TorchQLearningModel.t7")
local gridSize = opt.gridSize
local maxGames = opt.maxGames
local size = opt.renderSize
local backgroundColour = { opt.backGroundColours.blue, opt.backGroundColours.green, opt.backGroundColours.red } -- B,G,R
local env = CatchEnvironment(gridSize)
local function drawState(image, painter)
    painter.image = image
    local size = painter.window.size:totable()
    painter.refresh(size.width, size.height)
end

local displayImage = torch.Tensor(3, size, size)
local function processImage(state)
    state = state:view(10, 10)
    local display = image.scale(state, size, size, 'simple')
    for i = 1, displayImage:size(1) do
        displayImage[i]:copy(display)
        displayImage[i][torch.le(display, 0)] = backgroundColour[i]
    end
    return displayImage
end

local function sleep(n)
    os.execute("sleep " .. tonumber(n))
end

env.reset()
local currentState = env.observe()
local display = processImage(currentState)
local painter = image.display(display)
painter.window.windowTitle = 'TorchPlaysCatch'
drawState(display, painter)

local numberOfGames = 0
while numberOfGames < maxGames do
    -- The initial state of the environment.
    local isGameOver = false
    env.reset()
    local currentState = env.observe()
    drawState(processImage(currentState), painter)

    while (isGameOver ~= true) do
        -- Forward the current state through the network.
        local q = model:forward(currentState)
        -- Find the max index (the chosen action).
        local max, index = torch.max(q, 1)
        local action = index[1]
        local nextState, reward, gameOver = env.act(action)
        currentState = nextState
        isGameOver = gameOver
        drawState(processImage(currentState), painter)
        sleep(0.05)
    end
    collectgarbage()
end