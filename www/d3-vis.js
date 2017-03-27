function d3Simulate(path) {
  var svg = d3.select("svg"),
      width = +svg.attr("width"),
      height = +svg.attr("height");

  var color = d3.scaleOrdinal(d3.schemeCategory20);

  var simulation = d3.forceSimulation()
      .force("link", d3.forceLink()
          .id(function(d) { return d.id; })
          .strength(function(d) { return d.value; }))
      .force("charge", d3.forceManyBody())
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide().radius(
            function(d) { return d.radius + 0.5; }).iterations(2))
      .force("y", d3.forceY(0).strength(0.05))
      .force("x", d3.forceX(0).strength(0.05));

  var maxRadius = 50,
      padding = 6;

  var aspect = width / height,
      chart = d3.select('#chart');

  d3.select(window)
    .on("resize", function() {
      var targetWidth = chart.node().getBoundingClientRect().width;
      chart.attr("width", targetWidth);
      chart.attr("height", targetWidth / aspect);
    });

  d3.json(path, function(error, graph) {
    if (error) throw error;

    var link = svg.append("g")
        .attr("class", "links")
      .selectAll("line")
      .data(graph.links)
      .enter().append("line")
        .attr("stroke-width", function(d) { return Math.sqrt(d.value); });

    var node = svg.append("g")
        .attr("class", "nodes")
      .selectAll("circle")
      .data(graph.nodes)
      .enter().append("circle")
        .attr("r", function(d) { return d.radius; })
        .attr("fill", function(d) { return color(d.group); })
        .attr("group", function(d) { return d.group; })
        .attr("id", function(d) { return "node-" + d.id; })
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    node.append("title")
        .text(function(d) { return d.name; })

    simulation
        .nodes(graph.nodes)
        .on("tick", ticked);

    simulation.force("link")
        .links(graph.links);

    var zoom = d3.zoom()
        .scaleExtent([.5, 5])
        .translateExtent([[-width, -height], [width * 3, height * 3]])
        .on("zoom", zoomed);

    d3.select("button")
        .on("click", resetted);

    svg.call(zoom);

    function ticked() {
      link
          .attr("x1", function(d) { return d.source.x; })
          .attr("y1", function(d) { return d.source.y; })
          .attr("x2", function(d) { return d.target.x; })
          .attr("y2", function(d) { return d.target.y; });

      node
          .attr("cx", function(d) { return d.x; }) // = Math.max(d.radius, Math.min(width - d.radius, d.x)); })
          .attr("cy", function(d) { return d.y; }); //= Math.max(d.radius, Math.min(height - d.radius, d.y)); });
    }

    function zoomed() {
      link.attr("transform", d3.event.transform);
      node.attr("transform", d3.event.transform);
    }

    function resetted() {
      svg.transition()
          .duration(750)
          .call(zoom.transform, d3.zoomIdentity);
    }
  });

  function dragstarted(d) {
    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
    d3.event.sourceEvent.stopPropagation();
    d3.select(this).classed("dragging", true);
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(d) {
    d3.select(this).attr("cx", d.fx = d3.event.x).attr("cy", d.fy = d3.event.y);
    //d.fx = d3.event.x;
    //d.fy = d3.event.y;
  }

  function dragended(d) {
    d3.select(this).classed("dragging", false);
    if (!d3.event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }

}

d3Simulate("d3-forums-3-12.json");
