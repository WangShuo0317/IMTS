package com.imts.util;

import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.util.Base64;

public class CryptoUtil {
    
    private static final String ALGORITHM = "AES";
    private static final String SECRET_KEY = System.getenv().getOrDefault(
        "IMTS_CRYPTO_KEY", "IMTS-SECRET-KEY-32BYTES-LONG!!"
    );
    
    private static SecretKeySpec getSecretKey() {
        byte[] key = SECRET_KEY.getBytes(StandardCharsets.UTF_8);
        byte[] keyBytes = new byte[16];
        System.arraycopy(key, 0, keyBytes, 0, Math.min(key.length, 16));
        return new SecretKeySpec(keyBytes, ALGORITHM);
    }
    
    public static String encrypt(String strToEncrypt) {
        try {
            if (strToEncrypt == null || strToEncrypt.isEmpty()) {
                return null;
            }
            Cipher cipher = Cipher.getInstance(ALGORITHM);
            cipher.init(Cipher.ENCRYPT_MODE, getSecretKey());
            return Base64.getEncoder().encodeToString(
                cipher.doFinal(strToEncrypt.getBytes(StandardCharsets.UTF_8))
            );
        } catch (Exception e) {
            return null;
        }
    }
    
    public static String decrypt(String strToDecrypt) {
        try {
            if (strToDecrypt == null || strToDecrypt.isEmpty()) {
                return null;
            }
            Cipher cipher = Cipher.getInstance(ALGORITHM);
            cipher.init(Cipher.DECRYPT_MODE, getSecretKey());
            return new String(
                cipher.doFinal(Base64.getDecoder().decode(strToDecrypt)),
                StandardCharsets.UTF_8
            );
        } catch (Exception e) {
            return null;
        }
    }
}