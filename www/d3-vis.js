function getTransformation(transform) {
  // Create a dummy g for calculation purposes only. This will never
  // be appended to the DOM and will be discarded once this function 
  // returns.
  var g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  
  // Set the transform attribute to the provided string value.
  g.setAttributeNS(null, "transform", transform);
  
  // consolidate the SVGTransformList containing all transformations
  // to a single SVGTransform of type SVG_TRANSFORM_MATRIX and get
  // its SVGMatrix. 
  var matrix = g.transform.baseVal.consolidate().matrix;
  
  // Below calculations are taken and adapted from the private function
  // transform/decompose.js of D3's module d3-interpolate.
  var {a, b, c, d, e, f} = matrix;   // ES6, if this doesn't work, use below assignment
  // var a=matrix.a, b=matrix.b, c=matrix.c, d=matrix.d, e=matrix.e, f=matrix.f; // ES5
  var scaleX, scaleY, skewX;
  if (scaleX = Math.sqrt(a * a + b * b)) a /= scaleX, b /= scaleX;
  if (skewX = a * c + b * d) c -= a * skewX, d -= b * skewX;
  if (scaleY = Math.sqrt(c * c + d * d)) c /= scaleY, d /= scaleY, skewX /= scaleY;
  if (a * d < b * c) a = -a, b = -b, skewX = -skewX, scaleX = -scaleX;
  return {
    translateX: e,
    translateY: f,
    rotate: Math.atan2(b, a) * Math.PI/180,
    skewX: Math.atan(skewX) * Math.PI/180,
    scaleX: scaleX,
    scaleY: scaleY
  };
}


function d3Simulate(path) {
    var svg = d3.select("svg#force-directed"),
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
            .attr("node1", function(d) { return d.source; })
            .attr("node2", function(d) { return d.target; })
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
        if (!d3.event.active) 
            simulation.alphaTarget(0.3).restart();
        d3.event.sourceEvent.stopPropagation();
        d3.select(this).classed("dragging", true);
        d.fx = d.x;
        d.fy = d.y;
        d.last_x = d.x;
        d.last_y = d.y;
    }

    function dragged(d) {
        var transform = getTransformation(
            d3.select("svg g.nodes")
                .selectAll("circle")
                .attr("transform"));

        del_x = (d3.event.x - d.last_x) / transform.scaleX;
        del_y = (d3.event.y - d.last_y) / transform.scaleY;
        d.fx = d.last_x + del_x;
        d.fy = d.last_y + del_y;

        d3.select(this).attr("cx", d.fx).attr("cy", d.fy);
    }

    function dragended(d) {
        d3.select(this).classed("dragging", false);
        if (!d3.event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

}

d3Simulate("data/d3-forums-3-12.json");
