const tf = require('@tensorflow/tfjs-node');
const model = tf.sequential();

model.add(tf.layers.inputLayer({inputShape:[2]}));
model.add(tf.layers.dense({units:1}));

model.compile({
    loss:'meanSquaredError',
    optimizer:'sgd',
    metrics:['MAE']
});

var x_train =  [[2,3],[2,4],[1,1], [2,7]];
var y_train = [[5],[6],[2],[9]];
let data = numberGen(1000);

x_train = tf.tensor2d(data['x']);
y_train = tf.tensor2d(data['y']);
var x_val = [[1,2], [2,2], [4,5], [2,1], [5,5],[300,21]];
x_val = tf.tensor2d(x_val);
var y_val = [[3], [4], [9], [3],[10],[321]];
y_val = tf.tensor2d(y_val);

model.fit(x_train, y_train, {
    epochs:300,
    validationData: [x_val, y_val],
    callbacks: tf.node.tensorBoard('/tmp/fit_logs_1')
}).then(() => {
    model.save('file://./temp/my_model');
    let test = [
        [1,1],
        [2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]
    ]
    let oout = model.predict(tf.tensor2d(test));
    let data = oout.dataSync();
    test.forEach((val,inx) => {
        console.log(`${val[0]} + ${val[1]} = ${data[inx]}`);
    });
});

function numberGen(amount){
    let x_vals = [];
    let y_vals = [];
    for(var i =0; i < amount; i++) {
        let x1 = getRandomInt(10);
        let x2 = getRandomInt(10);
        x_vals.push([x1, x2]);
        y_vals.push([x1+x2]);
    }
    return {
        'x' : x_vals,
        'y' : y_vals
    }
}

function getRandomInt(max) {
    return Math.floor(Math.random() * Math.floor(max));
  }