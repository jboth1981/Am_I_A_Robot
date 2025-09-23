module.exports = {
  webpack: {
    configure: (webpackConfig) => {
      // Force webpack to use polling for file watching
      webpackConfig.watchOptions = {
        poll: 2000, // Poll every 10 seconds
        aggregateTimeout: 2000, // Wait 2 seconds after change before rebuilding
        ignored: /node_modules/
      };
      return webpackConfig;
    }
  },
  devServer: {
    client: {
      webSocketURL: 'wss://amiarobot.ca/ws',
      webSocketTransport: 'ws'
    },
    webSocketServer: 'ws',
    allowedHosts: 'all',
    hot: true,
    liveReload: false,
    headers: {
      'Access-Control-Allow-Origin': '*'
    },
    watchFiles: {
      paths: ['src/**/*'],
      options: {
        usePolling: true,
        interval: 10000
      }
    }
  }
};
