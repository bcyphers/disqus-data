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


function forceGraphSimulate(path, callback, initPos) {
    var svg = d3.select("svg#force-directed"),
        width = +svg.attr("width"),
        height = +svg.attr("height");

    var color = d3.scaleOrdinal(d3.schemeCategory20);

    var simulation = d3.forceSimulation()
        // links pull nodes together, strength proportional to correlation
        .force("link", d3.forceLink()
            .id(function(d) { return d.id; })
            // desired length is proportional to 
            .distance(function(d) { return 100 * Math.pow(1 - d.value, 1); })
            .strength(function(d) { return d.value; }))
        .force("charge", d3.forceManyBody())
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide()
            .radius(function(d) { return d.radius + 0.5; })
            .iterations(2))
        .force("y", d3.forceY(0).strength(0.05))
        .force("x", d3.forceX(0).strength(0.05));
  
    var maxRadius = 50,
        padding = 6;

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
            .attr("fill", "f2f2f2")
            .attr("id", function(d) { return "node-" + d.id; });

        if (initPos != null) {
            node.call(function(d) { 
                d.x = initPos["node-" + d.id].x; 
                d.y = initPos["node-" + d.id].y; });
        }

        node.call(d3.drag()
                .on("start", dragStarted)
                .on("drag", dragged)
                .on("end", dragEnded));

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

        svg.call(zoom);

        // hook up event handlers after everything else is done
        callback();

        function ticked() {
            link
                .attr("x1", function(d) { return d.source.x; })
                .attr("y1", function(d) { return d.source.y; })
                .attr("x2", function(d) { return d.target.x; })
                .attr("y2", function(d) { return d.target.y; });

            node
                .attr("cx", function(d) { return d.x; })
                .attr("cy", function(d) { return d.y; });
        }

        function zoomed() {
            link.attr("transform", d3.event.transform);
            node.attr("transform", d3.event.transform);
        }
    });

    function dragStarted(d) {
        if (!d3.event.active) 
            simulation.alphaTarget(0.3).restart();
        d3.event.sourceEvent.stopPropagation();
        d3.select(this).classed("dragging", true);
        d.fx = d.x;
        d.fy = d.y;
        d.lastX = d.x;
        d.lastY = d.y;
    }

    function dragged(d) {
        var transform = getTransformation(
            d3.select("svg g.nodes")
                .selectAll("circle")
                .attr("transform"));

        del_x = (d3.event.x - d.lastX) / transform.scaleX;
        del_y = (d3.event.y - d.lastY) / transform.scaleY;
        d.fx = d.lastX + del_x;
        d.fy = d.lastY + del_y;

        d3.select(this).attr("cx", d.fx).attr("cy", d.fy);
    }

    function dragEnded(d) {
        d3.select(this).classed("dragging", false);
        if (!d3.event.active) 
            simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}
