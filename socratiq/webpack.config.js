const path = require('path');

module.exports = {
  entry: './src_shadow/js/index.js', // Your entry point
  mode: 'development',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.js',
    publicPath: '/dist/', // Ensure the bundle is served from the correct path for HMR
  },
  module: {
    rules: [
      {
        test: /\.css$/i,
        use: ['to-string-loader', 'css-loader'],
      },
      {
        test: /\.html$/i,
        use: ['html-loader'],
      },
      {
        test: /\.md$/,
        use: ['html-loader', 'markdown-loader'],
      },
      {
        test: /\.worker\.js$/,
        use: { loader: 'worker-loader' },
      },
    ],
  },
  optimization: {
    moduleIds: 'named',
    chunkIds: 'named'
  },
  devServer: {
    static: {
      directory: path.join(__dirname, 'test_website/_book'), // Serve index.html from this directory
    },
    hot: true, // Enable HMR
    open: true, // Open the browser on start
    compress: true,
    port: 8080, // You can change the port if needed
    devMiddleware: {
      publicPath: '/dist/', // Ensure Webpack serves bundle.js correctly for HMR
    },
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin'
    }
  },
  resolve: {
    fallback: {
      "fs": false,
      "path": false,
      "crypto": false
    },
    modules: ['node_modules'],
    extensions: ['.js', '.jsx', '.json']
  }
};
