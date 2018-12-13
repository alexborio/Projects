package com.example.android.btunmute;

import android.app.ActivityManager;
import android.app.Service;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothProfile;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.ServiceConnection;
import android.media.AudioManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Looper;
import android.os.Message;
import android.os.Messenger;
import android.os.RemoteException;
import android.os.SystemClock;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.Switch;
import android.widget.CompoundButton;
import android.widget.CompoundButton.OnCheckedChangeListener;
import android.widget.Toast;
import android.content.Context;
import android.bluetooth.BluetoothAdapter;

import java.util.Set;


public class BtUnMute extends AppCompatActivity {

    private Switch onOffSwitch;
    private CheckBox statusBox;
    private AudioManager mAudioManager;
    private EditText mDeviceName;
    private boolean mIsBound;
    private Messenger mService;
    public static final int START = 1;
    public static final int STOP = 2;
    Set<BluetoothDevice> pairedDevices;

    Message msg;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        setContentView(R.layout.activity_bt_unmute);
        mDeviceName = (EditText) findViewById(R.id.deviceName);
        onOffSwitch = (Switch) findViewById(R.id.BtUnMuteOnOff);
        if (mService != null)
            onOffSwitch.setChecked(true);

        Intent broadcast1 = new Intent("getting_data");
        broadcast1.putExtra("value", mDeviceName.getText().toString());
        sendBroadcast(broadcast1);

        ActivityManager manager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);


        if (isMyServiceRunning(BkgService.class))
            onOffSwitch.setChecked(true);

        onOffSwitch.setOnCheckedChangeListener(new OnCheckedChangeListener() {

            @Override
            public void onCheckedChanged(CompoundButton buttonView,
                                         boolean isChecked) {

                CharSequence text = "";
                int duration = Toast.LENGTH_SHORT;

                // Register for broadcasts when a device is discovered.
                HandlerThread handlerThread = new HandlerThread("ht");
                handlerThread.start();
                Looper looper = handlerThread.getLooper();
                final Handler handler = new Handler(looper);

                if (isChecked) {
                    text = "BtUnMute is on! Awaiting a device connection...";
                    Context context = getApplicationContext();
                    Intent intent = new Intent(context, BkgService.class);
                    intent.putExtra("getDeviceName", mDeviceName.getText().toString());
                    msg = Message.obtain(null, BkgService.START, 0, 0);

                    startService(intent);
                    doBindService();

                    if (mService != null)
                        SendMsgToService();

                    Toast toast = Toast.makeText(context, text, duration);
                    toast.show();

                } else {

                    Intent broadcast = new Intent("SERVICE_STATE");
                    broadcast.putExtra("SERVICE_STATUS", "STOP");
                    sendBroadcast(broadcast);
                }
            }
        });

        mDeviceName.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {

            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {

                Intent broadcast1 = new Intent("getting_data");
                broadcast1.putExtra("value", mDeviceName.getText() + "");
                sendBroadcast(broadcast1);

            }

            @Override
            public void afterTextChanged(Editable s) {

            }
        });
    }


    public void setDeviceList(Set<BluetoothDevice> pairedDevices)
    {
        this.pairedDevices = pairedDevices;
    }

    private boolean isMyServiceRunning(Class<?> serviceClass) {
        ActivityManager manager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
        for (ActivityManager.RunningServiceInfo service : manager.getRunningServices(Integer.MAX_VALUE)) {
            if (serviceClass.getName().equals(service.service.getClassName())) {
                return true;
            }
        }
        return false;
    }


    private ActivityManager.RunningServiceInfo getMyServiceRunning(Class<?> serviceClass) {
        ActivityManager manager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
        for (ActivityManager.RunningServiceInfo service : manager.getRunningServices(Integer.MAX_VALUE)) {
            if (serviceClass.getName().equals(service.service.getClassName())) {
                return service;
            }
        }
        return null;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_bt_unmute, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();
        int itemThatWasClickedId = item.getItemId();
        //noinspection SimplifiableIfStatement
        if (id == R.id.BtUnMuteOnOff && onOffSwitch.isChecked()) {
            statusBox.setChecked(true);
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    private ServiceConnection mConnection = new ServiceConnection() {
        public void onServiceConnected(ComponentName className, IBinder service) {
            // This is called when the connection with the service has been
            // established, giving us the service object we can use to
            // interact with the service.  Because we have bound to a explicit
            // service that we know is running in our own process, we can
            // cast its IBinder to a concrete class and directly access it.
            mService = new Messenger(service);
            mIsBound = true;
            SendMsgToService();
            Context context = getApplicationContext();
            // Tell the user about this for our demo.
            Toast.makeText(context, R.string.local_service_connected, Toast.LENGTH_SHORT).show();
        }


        public void onServiceDisconnected(ComponentName className) {
            // This is called when the connection with the service has been
            // unexpectedly disconnected -- that is, its process crashed.
            // Because it is running in our same process, we should never
            // see this happen.
            mService = null;
            Context context = getApplicationContext();
            Toast.makeText(context, R.string.local_service_disconnected, Toast.LENGTH_SHORT).show();
        }
    };

    private void doBindService() {
        // Establish a connection with the service.  We use an explicit
        // class name because we want a specific service implementation that
        // we know will be running in our own process (and thus won't be
        // supporting component replacement by other applications).
        Intent intent = new Intent(this, BkgService.class);
        intent.putExtra("getDeviceName", mDeviceName.getText().toString());

        bindService(intent, mConnection, Context.BIND_AUTO_CREATE);
        mIsBound = true;
    }

    private void doUnbindService() {

        if (mIsBound) {
            // Detach our existing connection.
            unbindService(mConnection);
            mIsBound = false;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        doUnbindService();
    }

    private void SendMsgToService() {
        try {
            mService.send(msg);
        } catch (RemoteException e) {
            e.printStackTrace();
        }
    }

    private void HandlerWait(Handler handler, int millis) {

        try {
            handler.wait(millis);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }
}
