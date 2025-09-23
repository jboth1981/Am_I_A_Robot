module.exports = {
  webpack: {
    configure: (webpackConfig, { env, paths }) => {
      // Enable more detailed webpack logging
      if (env === 'development') {
        webpackConfig.stats = 'verbose';
      }
      // Force webpack to use polling for file watching
      webpackConfig.watchOptions = {
        poll: 2000, // Poll every 2 seconds
        aggregateTimeout: 2000, // Wait 2 seconds after change before rebuilding
        ignored: /node_modules/
      };
      
      // Fix NODE_ENV conflicts by ensuring consistent environment
      const DefinePlugin = webpackConfig.plugins.find(
        plugin => plugin.constructor.name === 'DefinePlugin'
      );
      if (DefinePlugin) {
        DefinePlugin.definitions['process.env.NODE_ENV'] = JSON.stringify(process.env.NODE_ENV || 'development');
      }
      
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
        interval: 2000
      }
    }
  }
};
