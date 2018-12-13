package com.example.android.btunmute;

import android.app.Activity;
import android.app.Notification;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.ContentObserver;
import android.media.AudioManager;
import android.media.AudioRouting;
import android.media.MediaRouter;
import android.media.VolumeProvider;
import android.net.rtp.AudioStream;
import android.os.Binder;
import android.os.Handler;
import android.os.IBinder;
import android.os.Message;
import android.os.Messenger;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Toast;
import android.provider.Settings;


import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Created by Alex on 9.6.2017.
 */

public class BkgService extends Service {
    private NotificationManager mNM;

    private String mDeviceName;
    // Unique Identification Number for the Notification.
    // We use it on Notification start, and to cancel it.
    private int NOTIFICATION = R.string.local_service_started;
    private AudioManager mAudioManager;
    public boolean isRunning;

    public static int NEW_DEVICE = 3;
    public static final int START = 1;
    public static final int STOP = 2;

    BluetoothAdapter mBluetoothAdapter;
    Set<BluetoothDevice> pairedDevices;

    final Messenger messenger = new Messenger( new IncomingHandler() );
    /**
     * Class for clients to access.  Because we know this service always
     * runs in the same process as its clients, we don't need to deal with
     * IPC.
     */
    public class LocalBinder extends Binder {
        BkgService getService() {
            return BkgService.this;
        }
    }

    private Notification makeNotification(){
        // build your foreground notification here
        // In this sample, we'll use the same text for the ticker and the expanded notification
        CharSequence text = getText(R.string.local_service_started);

        // The PendingIntent to launch our activity if the user selects this notification
        PendingIntent contentIntent = PendingIntent.getActivity(this, 0,
                new Intent(this, BtUnMute.class), 0);
        // Set the info for the views that show in the notification panel.
        Notification notification = new Notification.Builder(this)
                .setSmallIcon(R.mipmap.ic_launcher_bt)  // the status icon
                .setTicker(text)  // the status text
                .setWhen(System.currentTimeMillis())  // the time stamp
                .setContentTitle(getText(R.string.local_service_label))  // the label of the entry
                .setContentText(text)  // the contents of the entry
                .setContentIntent(contentIntent)  // The intent to send when the entry is clicked
                .build();
        // Send the notification.
        return notification;
    }
    @Override
    public void onCreate() {

        Context context = getApplicationContext();
        IntentFilter intentFilter = new IntentFilter();
        // set the custom action
        intentFilter.addAction("getting_data"); //Action is just a string used to identify the receiver as there can be many in your app so it helps deciding which receiver should receive the intent.

        IntentFilter filterConnected = new IntentFilter(BluetoothDevice.ACTION_ACL_CONNECTED);
        IntentFilter filterPaired = new IntentFilter(BluetoothDevice.EXTRA_DEVICE);
        IntentFilter filterStatus = new IntentFilter("SERVICE_STATE");

        context.registerReceiver(mReceiver, filterConnected, null, null);
        context.registerReceiver(mReceiver, filterPaired, null, null);
        context.registerReceiver(broadcastReceiver, intentFilter, null, null);
        context.registerReceiver(stopServiceReceiver, filterStatus, null, null);
        mNM = (NotificationManager)getSystemService(NOTIFICATION_SERVICE);
        mBluetoothAdapter = BluetoothAdapter.getDefaultAdapter();

        context.getContentResolver().registerContentObserver(
                android.provider.Settings.System.CONTENT_URI,
                false, mVolumeObserver);
        // Display a notification about us starting.  We put an icon in the status bar.
       // showNotification();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.i("BkgService", "Received start id " + startId + ": " + intent);
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        // Cancel the persistent notification.
       // mNM.cancel(NOTIFICATION);
        isRunning = false;
        // Tell the user we stopped.
        Toast.makeText(this, R.string.local_service_stopped, Toast.LENGTH_SHORT).show();
    }

    class IncomingHandler extends Handler {

        @Override
        public void handleMessage( Message msg ){
            switch( msg.what ){
                case START:
                    startService( new Intent( getApplicationContext(), BkgService.class ) );
                    Notification notification = makeNotification();
                    startForeground( NOTIFICATION, notification );
                    mNM.notify(NOTIFICATION, notification);
                    showNotification();

                    break;

                case STOP:
                    stopForeground( true );
                    stopSelf();
                    mNM.cancel(NOTIFICATION);

                    break;

                default:
                    super.handleMessage( msg );
            }
        }
    }

    public Set<BluetoothDevice> getDeviceList()
    {
        return pairedDevices;
    }

    @Override
    public IBinder onBind(Intent intent) {

        mDeviceName = intent.getStringExtra("getDeviceName");
        isRunning = true;
      //  Toast.makeText(getApplicationContext(), "BtUnMute: onBind called!! " + mDeviceName, Toast.LENGTH_SHORT).show();
        return messenger.getBinder();
    }

    // This is the object that receives interactions from clients.  See
    // RemoteService for a more complete example.
    // private final IBinder mBinder = new LocalBinder();

    /**
     * Show a notification while this service is running.
     */
    private void showNotification() {
        pairedDevices = mBluetoothAdapter.getBondedDevices();
        mAudioManager = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
    }
    BroadcastReceiver broadcastReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (action.equalsIgnoreCase("getting_data")) {
                mDeviceName = intent.getExtras().getString("value");
            }

        }
    };

    BroadcastReceiver stopServiceReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (action.equals("SERVICE_STATE") && intent.getExtras().getString("SERVICE_STATUS").equals("STOP")) {
                mNM.cancel(NOTIFICATION);
                stopForeground( true );
                stopSelf();
            }
        }
    };

    // Create a BroadcastReceiver for ACTION_FOUND.
    private final BroadcastReceiver mReceiver = new BroadcastReceiver() {
        public void onReceive(Context context, Intent intent) {

            String action = intent.getAction();
            pairedDevices = mBluetoothAdapter.getBondedDevices();


            if (BluetoothDevice.ACTION_ACL_CONNECTED.equals(action)) {

                BluetoothDevice device = intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE);
                String deviceName = device.getName();
                Toast.makeText(context, "Stored device name: " + mDeviceName,  Toast.LENGTH_SHORT).show();
                if (deviceName.equals(mDeviceName) ) {
                    MediaRouter mMediaRouter = (MediaRouter) getSystemService(Context.MEDIA_ROUTER_SERVICE);
                    Toast.makeText(context, "BtUnMute: Device connected!!", Toast.LENGTH_SHORT).show();
                    for (int i = 0; i < 2; i++) {
                        SystemClock.sleep(7000);
                        mAudioManager.setStreamVolume(AudioManager.STREAM_MUSIC, mAudioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC), AudioManager.FLAG_SHOW_UI);
                    }

                    Toast.makeText(context, "BtUnMute: Setting BT volume to max", Toast.LENGTH_SHORT).show();
                }
                else
                    Toast.makeText(context, "BtUnMute: No relevant device was discovered!!",  Toast.LENGTH_SHORT).show();
            }
        }
    };
    private ContentObserver mVolumeObserver = new ContentObserver(new Handler()) {
        @Override
        public void onChange(boolean selfChange) {
            super.onChange(selfChange);
            for (BluetoothDevice device : pairedDevices) {
                if(mDeviceName.equals(device.getName())){

                }
            }
            if (mAudioManager != null) {
                int volume = mAudioManager.getStreamVolume(AudioManager.STREAM_MUSIC);

            }
        }
    };
}