function HSVtoRGB(h, s, v) {
    var r, g, b, i, f, p, q, t;
    i = Math.floor(h * 6);
    f = h * 6 - i;
    p = v * (1 - s);
    q = v * (1 - f * s);
    t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }

    return {
        r : Math.round(r * 255),
        g : Math.round(g * 255),
        b : Math.round(b * 255),
    };
}

function buildTopicGraph(data) {
    var hue = 0, sat = 0.5, val = 0.95;
    var goldenRatio = 0.618033988749859;
    var topics = {};
    var totalWeight = Object.values(data['_baseline']).reduce(
            function(a, b) { return a + b; }, 0);
    var totalHeight = 500;

    // build a map from topics to top forums
    for (var t in data['_baseline']) { 

        // sort forums for this topic by score
        var list = Object.keys(data).map(function(f) { return [f, data[f][t]]; });
        list.sort(function(first, second) { return second[1] - first[1]; });
        topics[t] = { 
            weight: data['_baseline'][t] / totalWeight,
            forums: list,
        };
    }

    var topList = Object.keys(topics).map(function(t) { return [t, topics[t].weight]; });
    topList.sort(function(first, second) { return second[1] - first[1]; });
    topList = topList.map(function(t) { return t[0]; });

    for (var i = 0; i < 10; i++) {
        var html = "<li id='forum-" + i + "' class='row'> \
           <div class='col-2 topic-score'></div> \
           <div class='col forum-name'></div> \
         </li>";
        $('#topic-list ol').append(html);
    }
    
    for (var i = 0; i < topList.length; i++) {
        var t = topList[i];
        
        hue = (hue + goldenRatio) % 1;
        var rgb = HSVtoRGB(hue, 0.5, val);
        var cssSoft = 'rgb(' + rgb.r + ', ' + rgb.g + ', ' + rgb.b + ')';
        rgb = HSVtoRGB(hue, 0.8, val);
        cssBright = 'rgb(' + rgb.r + ', ' + rgb.g + ', ' + rgb.b + ')';

        $('div#topic-graph').append('<div class="topic-box" id="topic-' + i + '"></div>');
        $('div#topic-' + i).css({
            height: Math.round(topics[t].weight * totalHeight) + 'px',
            backgroundColor: cssSoft,
        });

        function setColor(color) { 
            return function() { $(this).css('background-color', color); } 
        }

        $('div#topic-' + i).hover(setColor(cssBright), setColor(cssSoft));

        var hoverFunc = (function(topic, forums, baseline) {
            // update information box with top forums for topic on hover
            return function(e) {
                $("#topic-list #topics-title").html('Top forums for topic "' + topic + '"');
                for (var j = 0; j < 10; j++) {
                    var forum = forums[j][0];
                    var value = forums[j][1] / baseline;
                    var num_dec = Math.max(2 - Math.floor(Math.log10(value)), 0);
                    value = value.toFixed(num_dec);

                    var name = details[forum].name;
                    if (details[forum].url != null) {
                        name = '<a href="' + details[forum].url + '">' + name + '</a>';

                    $("#topic-list #forum-" + j + " .topic-score").html(value);
                    $("#topic-list #forum-" + j + " .forum-name").html(name);
                }
            }
        })(t, topics[t].forums, data['_baseline'][t]);

        $('div#topic-' + i).hover(hoverFunc);
    }
}
