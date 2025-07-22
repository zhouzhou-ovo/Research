# 安装依赖（如果尚未安装）
install.packages("shiny")
install.packages("plotly")

# 运行应用
library(shiny)
library(plotly)

ui <- fluidPage(
  titlePanel("Weibull 分布密度函数动态演示"),
  
  sidebarLayout(
    sidebarPanel(
      sliderInput("shape", "Shape (α)", min = 0.1, max = 5, value = 1.5, step = 0.1),
      sliderInput("scale", "Scale (σ)", min = 0.1, max = 10, value = 1, step = 0.1)
    ),
    
    mainPanel(
      plotlyOutput("weibullPlot")
    )
  )
)

server <- function(input, output) {
  output$weibullPlot <- renderPlotly({
    x_vals <- seq(0, 10, length.out = 500)
    y_vals <- dweibull(x_vals, shape = input$shape, scale = input$scale)
    
    plot_ly(x = ~x_vals, y = ~y_vals, type = 'scatter', mode = 'lines',
            line = list(color = 'blue')) %>%
      layout(title = paste0("Weibull Density (α = ", input$shape,
                            ", σ = ", input$scale, ")"),
             xaxis = list(title = "x"),
             yaxis = list(title = "Density"))
  })
}

shinyApp(ui = ui, server = server)