package server;

import sneklms.Main;
import py4j.GatewayServer;

public class ServerApplication {

    public String compile(String src) {
        return Main.compileMain(src);
    }

    public static void main(String[] args) {
        ServerApplication app = new ServerApplication();
        // app is now the gateway.entry_point
        GatewayServer server = new GatewayServer(app);
        server.start();
    }
}
