import * as tf from "@tensorflow/tfjs";

const class_names = [
  "antelope",
  "badger",
  "bat",
  "bear",
  "bee",
  "beetle",
  "bison",
  "boar",
  "butterfly",
  "cat",
  "caterpillar",
  "chimpanzee",
  "cockroach",
  "cow",
  "coyote",
  "crab",
  "crow",
  "deer",
  "dog",
  "dolphin",
  "donkey",
  "dragonfly",
  "duck",
  "eagle",
  "elephant",
  "flamingo",
  "fly",
  "fox",
  "goat",
  "goldfish",
  "goose",
  "gorilla",
  "grasshopper",
  "hamster",
  "hare",
  "hedgehog",
  "hippopotamus",
  "hornbill",
  "horse",
  "hummingbird",
  "hyena",
  "jellyfish",
  "kangaroo",
  "koala",
  "ladybugs",
  "leopard",
  "lion",
  "lizard",
  "lobster",
  "mosquito",
  "moth",
  "mouse",
  "octopus",
  "okapi",
  "orangutan",
  "otter",
  "owl",
  "ox",
  "oyster",
  "panda",
  "parrot",
  "pelecaniformes",
  "penguin",
  "pig",
  "pigeon",
  "porcupine",
  "possum",
  "raccoon",
  "rat",
  "reindeer",
  "rhinoceros",
  "sandpiper",
  "seahorse",
  "seal",
  "shark",
  "sheep",
  "snake",
  "sparrow",
  "squid",
  "squirrel",
  "starfish",
  "swan",
  "tiger",
  "turkey",
  "turtle",
  "whale",
  "wolf",
  "wombat",
  "woodpecker",
  "zebra",
];

let model;

export const loadModel = async (url) => {
  model = await tf.loadGraphModel(url);
  return model;
};

export const predict = async (image) => {
  try {
    let tensor = tf.browser
      .fromPixels(image)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .expandDims(0);

    console.log(tensor.dataSync());

    const pred_probs = await model.predict(tensor);
    pred_probs = pred_probs.dataSync();
    console.log(pred_probs);
    const pred = tf.tensor1d(pred_probs).argMax().dataSync();
    //   class_names[pred];
    console.log(class_names[pred]);
    return class_names[pred];
  } catch (error) {
    console.log(error.message);
  }
};
