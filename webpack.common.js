const path = require("path");
const webpack = require('webpack')

// const environmentVariables = [
//   "BASE_URL",
// ];


module.exports = {
  entry: path.resolve(__dirname,  "client/index.js"),

  // plugins: [
  //   new webpack.EnvironmentPlugin(environmentVariables)
  // ],

  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
          options: {
            presets: ["@babel/preset-env", "@babel/preset-react"],
          },
        },
      },
    ],
  },


  output: {
    filename: 'bundle.js',
    // path: path.resolve(__dirname, 'dist'),
    path: path.resolve(__dirname, 'server/venv/static'),
  }
};