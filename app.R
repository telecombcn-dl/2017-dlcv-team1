#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#


required_packages <- c("shiny", "mxnet", "imager", "scales", "jpeg", "ggplot2", "readr", "png", "ggthemes")
sapply(required_packages, require, character.only = TRUE)


preproc.image <- function(image) {
  

  shape <- dim(image)
  short.edge <- min(shape[1:2])
  xx <- floor((shape[1] - short.edge) / 2)
  yy <- floor((shape[2] - short.edge) / 2) 
  croped <- crop.borders(image, xx, yy)
  resized <- resize(croped, 299, 299)
  array <- as.array(resized) * 256
  dim(array) <- c(299, 299, 3)

  preprocessedImage <- array - 128
  preprocessedImage <- array/128
  
  dim(preprocessedImage) <- c(299, 299, 3, 1)
  return(preprocessedImage)
}



picksource <- function(input){
  
  ImageFromUrl <- eventReactive(input$enter, {
    print(input$urlImage)
   
      image <- tempfile()
      download.file(input$urlImage, destfile = image, method = "auto")
      return(image)
    
  })
  
  
  src = if (input$tabs == "Upload Image") {
    if (is.null(input$Imagefile)) {
      if (input$goButton == 0 || is.null(ImageFromUrl())) {
        'image.jpg'
      } else {
        ImageFromUrl()
      }
    } else {
      input$Imagefile$datapath
    }
  } else {
    if (input$goButton == 0 || is.null(ImageFromUrl())) {
      if (is.null(input$Imagefile)) {
        'image.jpg'
      } else {
        input$Imagefile$datapath
      }
    } else {
      ImageFromUrl()
    }
  }
}


getCommonResults <- function(result){
  StringResult <- ""
  for (i in 1:5) {
    ResultInJ <- strsplit(result[i], " ")[[1]]
    for (j in 2:length(ResultInJ)) {
      StringResult <- paste(StringResult, ResultInJ[j])
    }
    StringResult <- paste(StringResult, "\n")
  }
  StringResult

}


# Define UI for application that draws a histogram
ui <- fluidPage(
  
  includeCSS('alike.css'),
  # Application title
  titlePanel("Image Classification using Convolutional Neural Networks"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      tabsetPanel(
        id = "tabs",
        tabPanel("Upload Image",
                 fileInput('Imagefile', '\n Upload an Image:')),
        tabPanel(
          "Use the URL",
          textInput("urlImage", "Enter an Image URL:", ""),
          actionButton("enter", "OK")
        )
      )
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      h3("Image to classify"),
      tags$hr(),
      imageOutput("imageloaded", height = "auto"),
      tags$hr(),
      h3("What does the image contains?"),
      tags$hr(),
      verbatimTextOutput("res"),
      plotOutput("ProbPlot")
    )
  )
)

# Define server logic 
server <- function(input, output) {
  

  
  if (!file.exists("model")) {
    download.file("http://data.dmlc.ml/mxnet/models/imagenet/inception-v3.tar.gz", destfile = "inception-v3.tar.gz")
    untar("inception-v3.tar.gz")
  }
  
  ImageNet_Model <- mx.model.load("model/Inception-7", iteration = 1)
  
  synsets <- read_lines("model/synset.txt")
  
  image <- NULL
  
  
  
  
  output$imageloaded = renderImage(list(src = picksource(input)), deleteFile = FALSE)
  
  
  
  output$res <- renderText({
    src <- picksource(input)
    
    im <- load.image(src)
    preprocessedImage <- preproc.image(im)
    prob <- predict(ImageNet_Model, X = preprocessedImage)
    max.idx <- order(prob[,1], decreasing = TRUE)[1:5]
    result <- synsets[max.idx]
    StringResult <- ""
    for (i in 1:5) {
      ResultInJ <- strsplit(result[i], " ")[[1]]
      for (j in 2:length(ResultInJ)) {
        StringResult <- paste(StringResult, ResultInJ[j])
      }
      StringResult <- paste(StringResult, "\n")
    }
    StringResult
  })
  
  
  
  output$res <- renderText({
    src <- picksource(input)
    
    im <- load.image(src)
    preprocessedImage <- preproc.image(im)
    prob <- predict(ImageNet_Model, X = preprocessedImage)
    max.idx <- order(prob[,1], decreasing = TRUE)[1:5]
    result <- synsets[max.idx]
    StringResult <- StringResult <- getCommonResults(result)
  })
  
  output$ProbPlot <- renderPlot({
   src <- picksource(input)
    
    im <- load.image(src)
    preprocessedImage <- preproc.image(im)
    prob <- predict(ImageNet_Model, X = preprocessedImage)
    max.idx <- order(prob[,1], decreasing = TRUE)[1:5]
    result <- synsets[max.idx]
    StringResult <- getCommonResults(result)
    
    
    StringResult1 <- unlist(strsplit(StringResult, split="\n"))
    StringResult1 <- data.frame(names = StringResult1, probability = sort(prob[,1], decreasing = TRUE)[1:5])
    ggplot(StringResult1) + geom_bar(aes(reorder(names, probability ), probability * 100), stat = "identity", fill = "cyan", alpha = 1/3) + coord_flip() +
      labs(x = "Names", y = "Probability (%)") + theme_solarized_2(light = FALSE) + theme_hc(bgcolor = "darkunica") +
      scale_colour_hc("darkunica")
  })
  
  
}

# Run the application 
shinyApp(ui = ui, server = server)