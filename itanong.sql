-- MySQL dump 10.13  Distrib 8.0.30, for Win64 (x86_64)
--
-- Host: localhost    Database: itanong
-- ------------------------------------------------------
-- Server version	8.0.30

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `category`
--

DROP TABLE IF EXISTS `category`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `category` (
  `category_id` int NOT NULL,
  `category_description` varchar(255) NOT NULL,
  PRIMARY KEY (`category_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `category`
--

LOCK TABLES `category` WRITE;
/*!40000 ALTER TABLE `category` DISABLE KEYS */;
INSERT INTO `category` VALUES (801,'Coconuts, Brazil nuts and cashew nuts, fresh or dried, whether or not shelled or peeled.'),(802,'Other nuts, fresh or dried, whether or not shelled or peeled.'),(803,'Bananas, including plantains, fresh or dried.'),(804,'Dates, figs, pineapples, avocados, guavas, mangoes and mangosteens, fresh or dried.'),(805,'Citrus fruit, fresh or dried.'),(806,'Grapes, fresh or dried.'),(807,'Melons (including watermelons) and papaws (papayas), fresh.'),(808,'Apples, pears and quinces, fresh.'),(809,'Apricots, cherries, peaches (including nectarines), plums and sloes, fresh.'),(810,'Other fruit, fresh.'),(811,'Fruit and nuts, uncooked or cooked by steaming or boiling in water, frozen, whether or not containing added sugar or other sweetening matter.'),(812,'Fruit and nuts provisionally preserved, but unsuitable in that state for immediate consumption.'),(813,'Fruit, dried, mixtures of nuts or dried fruits.'),(814,'Peel of citrus fruit or melons (including watermelons), fresh, frozen, dried or provisionally preserved in brine, in sulphur water or in other preservative solutions.');
/*!40000 ALTER TABLE `category` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `products`
--

DROP TABLE IF EXISTS `products`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `products` (
  `product_id` int NOT NULL,
  `category_id` int DEFAULT NULL,
  `rate_MFN` int DEFAULT NULL,
  `rate_ATIGA` int DEFAULT NULL,
  `product_description` varchar(255) DEFAULT NULL,
  `effective_year` int DEFAULT NULL,
  PRIMARY KEY (`product_id`),
  KEY `category_id` (`category_id`),
  CONSTRAINT `products_ibfk_1` FOREIGN KEY (`category_id`) REFERENCES `category` (`category_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `products`
--

LOCK TABLES `products` WRITE;
/*!40000 ALTER TABLE `products` DISABLE KEYS */;
INSERT INTO `products` VALUES (8011100,801,15,0,'Coconuts, Desiccated',2022),(8011200,801,10,0,'Coconuts, Fresh',2022),(8011910,801,10,0,'Coconuts, Young',2022),(8011990,801,10,0,'Coconuts, Other',2022),(8012100,801,7,0,'Brazil Nuts, In Shell',2022),(8012200,801,5,0,'Brazil Nuts, Shelled',2022),(8013100,801,3,0,'Cashew Nuts, In Shell',2022),(8013200,801,2,0,'Cashew Nuts, Shelled',2022),(8021100,802,3,0,'Almonds, In Shell',2022),(8021210,802,3,0,'Almonds, Shelled, Blanched',2022),(8021290,802,3,0,'Almonds, Shelled, Other',2022),(8022100,802,3,0,'Hazelnuts, In Shell',2022),(8022200,802,3,0,'Hazelnuts, Shelled',2022),(8023100,802,3,0,'Walnuts, In Shell',2022),(8023200,802,3,0,'Walnuts, Shelled',2022),(8024100,802,3,0,'Chestnuts, In Shell',2022),(8024200,802,3,0,'Chestnuts, Shelled',2022),(8025100,802,3,0,'Pistachios, In Shell',2022),(8025200,802,3,0,'Pistachios, Shelled',2022),(8026100,802,3,0,'Macadamia Nuts, In Shell',2022),(8026200,802,3,0,'Macadamia Nuts, Shelled',2022),(8027000,802,3,0,'Kola Nuts',2022),(8028000,802,3,0,'Areca Nuts',2022),(8029100,802,3,0,'Pine Nuts, In Shell',2022),(8029200,802,3,0,'Pine Nuts, Shelled',2022),(8029900,802,3,0,'Other Nuts',2022),(8031010,803,15,0,'Plantains, fresh',2022),(8031020,803,15,0,'Plantains, dried',2022),(8039010,803,15,0,'Lady\'s finger banana',2022),(8039020,803,15,0,'Cavendish banana (Musa acuminata)',2022),(8039030,803,15,0,'Chestnut banana (hybrid of Musa acuminata and Musa balbisiana, cultivar Berangan)',2022),(8039090,803,15,0,'Other',2022),(8041000,804,3,0,'Dates',2022),(8042000,804,3,0,'Figs',2022),(8043000,804,10,0,'Pineapples',2022),(8044000,804,15,0,'Avocados',2022),(8045010,804,15,0,'Guavas',2022),(8045021,804,15,0,'Mangoes, fresh',2022),(8045022,804,15,0,'Mangoes, dried',2022),(8045030,804,15,0,'Mangosteens',2022),(8051010,805,10,0,'Oranges, fresh',2022),(8051020,805,10,0,'Oranges, dried',2022),(8052100,805,10,0,'Mandarins (including tangerines and satsumas)',2022),(8052200,805,10,0,'Clementines',2022),(8052900,805,10,0,'Other mandarins, clementines, wilkings and similar citrus hybrids',2022),(8054000,805,7,0,'Grapefruit and pomelos',2022),(8055010,805,10,0,'Lemons (Citrus limon, Citrus limonum)',2022),(8055020,805,10,0,'Limes (Citrus aurantifolia, Citrus latifolia)',2022),(8059000,805,10,0,'Other citrus fruit',2022),(8061000,806,7,0,'Grapes, fresh',2022),(8062000,806,3,0,'Grapes, dried',2022),(8071100,807,15,0,'Watermelons',2022),(8071900,807,15,0,'Other melons',2022),(8072000,807,15,0,'Papaws (papayas)',2022),(8081000,808,7,0,'Apples',2022),(8083000,808,7,0,'Pears',2022),(8084000,808,7,0,'Quinces',2022),(8091000,809,7,0,'Apricots',2022),(8092100,809,7,0,'Sour cherries (Prunus cerasus)',2022),(8092900,809,7,0,'Other cherries',2022),(8093000,809,7,0,'Peaches, including nectarines',2022),(8094010,809,7,0,'Plums',2022),(8094020,809,7,0,'Sloes',2022),(8101000,810,15,0,'Strawberries',2022),(8102000,810,7,0,'Raspberries, blackberries, mulberries and loganberries',2022),(8103000,810,7,0,'Black, white or red currants and gooseberries',2022),(8104000,810,7,0,'Cranberries, bilberries and other fruits of the genus Vaccinium',2022),(8105000,810,7,0,'Kiwifruit',2022),(8106000,810,10,0,'Durians',2022),(8107000,810,10,0,'Persimmons',2022),(8109010,810,10,0,'Longans; Mata Kucing',2022),(8109020,810,10,0,'Lychees',2022),(8109030,810,10,0,'Rambutan',2022),(8109040,810,10,0,'Langsat (Lanzones)',2022),(8109050,810,10,0,'Jackfruit (including Cempedak and Nangka)',2022),(8109060,810,10,0,'Tamarinds',2022),(8109070,810,10,0,'Starfruit',2022),(8109091,810,10,0,'Salacca (snake fruit)',2022),(8109092,810,10,0,'Dragon fruit',2022),(8109093,810,10,0,'Sapodilla (ciku fruit)',2022),(8111000,811,15,0,'Strawberries, frozen',2022),(8112000,811,7,0,'Raspberries, blackberries, mulberries, loganberries, black, white or red currants and gooseberries, frozen',2022),(8119000,811,7,0,'Other fruit and nuts, frozen',2022),(8121000,812,3,0,'Cherries, provisionally preserved',2022),(8129010,812,15,0,'Strawberries, provisionally preserved',2022),(8129090,812,7,0,'Other fruit and nuts, provisionally preserved',2022),(8131000,813,7,0,'Dried apricots',2022),(8132000,813,7,0,'Dried prunes',2022),(8133000,813,7,0,'Dried apples',2022),(8134010,813,7,0,'Dried longans',2022),(8134020,813,7,0,'Dried tamarinds',2022),(8134090,813,7,0,'Other dried fruit',2022),(8135010,813,7,0,'Mixtures of nuts or dried fruits of which cashew nuts or Brazil nuts predominate by weight',2022),(8135020,813,7,0,'Mixtures of nuts or dried fruits of which other nuts predominate by weight',2022),(8135030,813,7,0,'Mixtures of nuts or dried fruits of which dates predominate by weight',2022),(8135040,813,7,0,'Mixtures of nuts or dried fruits of which avocados or oranges or mandarins (including tangerines and satsumas) predominate by weight',2022),(8135090,813,7,0,'Other mixtures of nuts or dried fruits',2022),(8140000,814,20,0,'Peel of citrus fruit or melons (including watermelons), fresh, frozen, dried or provisionally preserved in brine, in sulphur water or in other preservative solutions',2022);
/*!40000 ALTER TABLE `products` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-05-23 11:19:22
