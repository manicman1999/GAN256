function perform() {
    
    //Fill Array With Values
    noise = [[], [], [], []];
    
    //4096 values between -1 and 1, uniform
    for(i = 0; i < 4096; i++) {
        noise[0][i] = Math.random() * 2 - 1;
        noise[1][i] = Math.random() * 2 - 1;
        noise[2][i] = Math.random() * 2 - 1;
        noise[3][i] = Math.random() * 2 - 1;
    }
    
    //Convert to tfjs tensor
    inp = tf.tensor(noise)
    
    //Get Output
    if(model) {
        output = model.predict(inp).dataSync();
    } else {
        return;
    }
    
    //Make Image
    drawn = [];
    
    index = 0;
    
    for(im = 0; im < 4; im++) {
        for(row = 0; row < 256; row++) {
            
            onrow = row + ((im % 2) * 256)
            
            //Make New Rows
            if(im < 2)
                drawn[onrow] = []
            
            for(col = 0; col < 256; col++) {
                
                oncol = col + (Math.floor(im / 2) * 256)
                
                //Make New Col
                drawn[onrow][oncol] = []
                
                for(chan = 0; chan < 3; chan++) {
                    
                    //Fill Value With Value Of output[index]
                    drawn[onrow][oncol][chan] = output[index];
                    index++;
                    
                }
            }
        }
    }
    
    tf.toPixels(tf.tensor(drawn), c);
    
    
}