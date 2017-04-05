d3.json("data/forum-correlations-beta-small.json", function(data) {
  var cell_size = 4;
  var nvar = data["index"].length;
  var h = cell_size * nvar;
  var w = h;
  var pad = {
    left: 20,
    top: 20,
    right: 20,
    bottom: 20
  };

  var innerPad = 5;
  var totalh = h + pad.top + pad.bottom;
  var totalw = w + pad.left + pad.right;
    
  var svg = d3.select("div#matrix")
      .append("svg")
      .attr("id", "matrix-plot")
      .attr("height", totalh)
      .attr("width", totalw);

  var corrplot = svg.append("g")
    .attr("id", "corrplot")
    .attr("transform", 
          "translate(" + pad.left + "," + pad.top + ")");

  var corXscale = d3.scaleBand()
    .domain(d3.range(nvar))
    .rangeRound([0, w]);

  var corYscale = d3.scaleBand()
    .domain(d3.range(nvar))
    .rangeRound([0, h]);

  var corZscale = d3.scaleLinear()
    .domain([-1, 0, 1])
    .range(["darkslateblue", "white", "crimson"]);

  var corr = [];
  
  var i, j;
  for (i in data["data"]) {
    for (j in data["data"][i]) {
      corr.push({ row: i, col: j, value: data["data"][i][j] });
    }
  }

  // Update the text showing the names of the two forums and the correlation
  var mouseOver = function(d) { 
    d3.select(this).attr("stroke", "black"); 

    var getTextX = function() { 
      var mult = -1;
      if (d.col < nvar / 2) { mult = 1; }
      return corXscale(d.col) + mult * 30;
    }
    
    var getTextY = function() {
      var mult = -1;
      if (d.row < nvar / 2) { mult = 1; }
      return corYscale(d.row) + (mult + 0.35) * 20;
    }

    var bounds;
    function getBounds(selection) {
        selection.each(function(d) { bounds = this.getBBox(); });
    }

    corrplot.append("text")
      .attr("id", "corrtext")
      .text(d3.format(".2f")(d.value))
      .attr("x", getTextX)
      .attr("y", getTextY)
      .attr("fill", "black")
      .attr("dominant-baseline", "middle")
      .attr("text-anchor", "middle")
      // get bounding box to allow text background
      .call(getBounds);

    // insert background rect to make text readable
    corrplot.insert("rect", "text")
      .attr("id", "corrtext-bg")
      .attr("x", bounds.x - 3)
      .attr("y", bounds.y - 1)
      .attr("rx", 2).attr("ry", 2)  // rounded corners
      .attr("width", bounds.width + 6)
      .attr("height", bounds.height + 2)
      .attr("stroke", "black")
      .attr("fill", "white");

    corrplot.append("text")
      .attr("class", "corrlabel")
      .attr("x", corXscale(d.col))
      .attr("y", h + pad.bottom)
      .text(details[data["index"][d.col]].name)
      .attr("dominant-baseline", "middle")
      .attr("text-anchor", "middle");

    corrplot.append("text")
      .attr("class", "corrlabel")
      .attr("x", -pad.left * 0.5)
      .attr("y", corYscale(d.row))
      .text(details[data["index"][d.row]].name)
      .attr("dominant-baseline", "middle")
      .attr("text-anchor", "end");
  }

  var mouseOut = function() {
    d3.selectAll("text.corrlabel").remove();
    d3.selectAll("text#corrtext").remove();
    d3.selectAll("rect#corrtext-bg").remove();
    return d3.select(this).attr("stroke", "none");
  }

  var cells = corrplot.append("g")
    .attr("id", "cells")
    .selectAll("empty")
    .data(corr)
    .enter().append("rect")
      .attr("class", "cell")
      .attr("x", function(d) { return corXscale(d.col); })
      .attr("y", function(d) { return corYscale(d.row); })
      .attr("width", corXscale.bandwidth())
      .attr("height", corYscale.bandwidth())
      .attr("fill", function(d) { return corZscale(d.value); })
      .attr("stroke", "none")
      .attr("stroke-width", 2)
      .on("mouseover", mouseOver)
      .on("mouseout",  mouseOut)
      .on("click", function(d) { return drawScatter(d.col, d.row); });
});
